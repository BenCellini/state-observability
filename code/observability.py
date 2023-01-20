
import numpy as np
import scipy
import pynumdiff
import cvxpy

# import copy
# from simulator import LinearSystemSimulator
# import matplotlib.pyplot as plt


class SingleStateObservability:
    def __init__(self, O, noise_threshold=None, min_sing_val=None):
        """ Find the observability of a single state(s) given an observability matrix O.

            Inputs
                O:  observability matrix
        """

        self.O = O  # original provided O matrix
        self.O_removed_noise = []  # O after noise removal
        self.remove_noise_from_O(noise_threshold=noise_threshold, min_sing_val=min_sing_val)

    def remove_noise_from_O(self, noise_threshold=0.01, min_sing_val=0.1):
        """ Remove noise from O using rank truncation.
            Steps:
                1. Find SVD of O
                2. Choose the smallest singular value that is "important"
                3. Reconstruct O after truncating rank to that singular value

            Inputs
                noise_threshold:    noise removal threshold
                min_sing_val:       for rank truncation to remove noise
        """

        if noise_threshold is None:
            # O_as_vector = np.squeeze(np.reshape(self.O, (self.O.shape[0] * self.O.shape[1], 1)))
            # noise_threshold = np.percentile(O_as_vector, 5)  # default is the 5th percentile of all elements of O
            noise_threshold = 0.01

        if min_sing_val is None:
            min_sing_val = 0.1

        U, E, V = np.linalg.svd(self.O)
        truncated_rank = np.where(E > min_sing_val)[0][-1] + 1

        Emat = np.zeros([truncated_rank, V.shape[0]])
        Emat[0:len(E[0:truncated_rank]), 0:len(E[0:truncated_rank])] = np.diag(E[0:truncated_rank])

        U_truncated = U[:, 0:truncated_rank]

        reconstructed = U_truncated @ Emat @ V
        reconstructed[np.where(np.abs(reconstructed) < noise_threshold)] = 0
        self.O_removed_noise = reconstructed

    def get_subset_O(self, state, noise_removed=True):
        """ Pick the subset rows of O that correspond to a given state.

            Inputs
                state:          state index of interest (start at 1)
                noise_removed:  (boolean) use O with noise removed if True

            Outputs
                O_rows_with_state_of_interest: rows of O that have a nonzero entry for the state of interest
                O_other_rows: rows of O that do not have a nonzero entry for the state of interest
        """

        state = int(state - 1)  # subtract 1 to index from 0

        # Use original O or O with noise removed
        if noise_removed:  # removed noise
            O = self.O_removed_noise.copy()
        else:  # original O
            O = self.O.copy()

        # Logical vector with true value for state of interest
        state_vector_of_interest = np.zeros([1, O.shape[1]])
        state_vector_of_interest[0, state] = 1

        # Get subset O of corresponding to the state of interest
        O_rows_with_state_of_interest = []
        O_other_rows = []
        for row in range(O.shape[0]):  # each row in O
            if O[row, state] != 0:  # row contains state of interest
                O_rows_with_state_of_interest.append(O[row, :])
            else:  # row does not contain state of interest
                O_other_rows.append(O[row, :])

        O_rows_with_state_of_interest = np.array(O_rows_with_state_of_interest)
        O_other_rows = np.array(O_other_rows)

        return O_rows_with_state_of_interest, O_other_rows

    def minimum_subset_O(self, state, rows=None, gamma=1e-10):
        """ Pick the subset rows of O that correspond to a given state.

            Inputs
                state:      state index of interest (start at 1)
                rows:       row indices to check
                gamma:      regularizer value for optimization

            Outputs
                O_subset: subset of rows of O needed for the state of interest to be observable
                rows_of_O_other_needed_for_observability: rows of O that have a nonzero entry needed for
                the state of interest to be observable
        """

        if gamma is None:
            gamma = 1e-10

        # Logical vector with true value for state of interest
        state = int(state)  # subtract 1 to index from 0
        state_vector_of_interest = np.zeros([1, self.O.shape[1]])
        state_vector_of_interest[0, state - 1] = 1

        # Separate O into rows that contain the state of interest & rows that don't
        O_rows_with_state_of_interest, O_other_rows = self.get_subset_O(state, noise_removed=True)

        # Set rows to test
        if rows is None:
            rows = np.arange(0, O_rows_with_state_of_interest.shape[0], 1)  # all rows
        else:
            rows = np.array([rows - 1])  # zero index for given rows

        # Design cost function to minimum subset of rows that don't contain the state of interest
        # that are needed for the state to be observable
        v = cvxpy.Variable((1, O_other_rows.shape[0]))
        m = cvxpy.Variable(1)  # needed to get loss function to actually go to zero
        regularizer = cvxpy.norm1(v)

        # For each row that contains the state of interest, find the other rows needed for the state to be observable
        rows_of_O_other_needed_for_observability = []
        O_subset = []
        O_subset_augmented = []
        observable = []
        for r in range(len(rows)):
            # Set up loss function
            L = O_rows_with_state_of_interest[rows[r]:rows[r] + 1] - \
                cvxpy.multiply(m, state_vector_of_interest) - \
                cvxpy.matmul(v, O_other_rows)

            Loss = cvxpy.norm2(L) + gamma * regularizer  # ensures we pick a sparse collection

            # Minimize loss function using convex optimization
            obj = cvxpy.Minimize(Loss)
            prob = cvxpy.Problem(obj)
            prob.solve(solver='MOSEK')

            # Pull out required rows
            rows_needed = np.where(np.abs(v.value).T > 0.001)[0]
            rows_of_O_other_needed_for_observability.append(rows_needed)

            # Create new subset of O needed for the state to be observable
            subset = np.vstack([O_rows_with_state_of_interest[rows[r]]] + [O_other_rows[i] for i in rows_needed])
            O_subset.append(subset)

            # Augment subset with state vector
            augmented = np.vstack((subset, state_vector_of_interest))
            O_subset_augmented.append(augmented)

            # Check observability by comparing rank of subset & augmented subset
            check = np.linalg.matrix_rank(augmented) - np.linalg.matrix_rank(subset)
            if check == 0:  # rank doesn't change
                observable.append(True)
            else:
                observable.append(False)

        return O_subset, rows_of_O_other_needed_for_observability, O_subset_augmented, observable

    def get_subset_iterate_O(self, state, endrow, gamma=0.1, noise_removed=True):
        """ Pick the subset rows of O that correspond to a given state.

            Inputs
                state:      state index of interest (start at 1)
                row:        index of row in O to consider (up to this row)
                gamma:      regularizer value for optimization

            Outputs
                O_subset: subset of rows of O
        """
        state = int(state - 1)  # subtract 1 to index from 0

        # Use original O or O with noise removed
        if noise_removed:  # removed noise
            O = self.O_removed_noise.copy()
        else:  # original O
            O = self.O.copy()

        # Logical vector with true value for state of interest
        state_vector_of_interest = np.zeros([1, O.shape[1]])
        state_vector_of_interest[0, state] = 1

        # Set up loss function
        v = cvxpy.Variable([1, O[endrow:, :].shape[0]])
        L = np.array([[1, 0, 0]]) - cvxpy.matmul(v, O[endrow:, :])
        regularizer = cvxpy.norm1(v)
        Loss = cvxpy.norm2(L) + gamma * regularizer  # ensures we pick a sparse collection

        # Optimize
        obj = cvxpy.Minimize(Loss)
        prob = cvxpy.Problem(obj)
        prob.solve(solver='MOSEK')

        # Get subset of O
        O_subset = O[endrow:, :][np.where(np.abs(v.value) > 0.0001)[1]]

        return O_subset

    def get_subset_states_O(self, states, endrow=None):
        """ Pick the subset rows of O that correspond to a given state.

            Inputs
                states:     state indices of interest (start at 1)
                row:        index of row in O to consider
                gamma:      regularizer value for optimization

            Outputs
                O_subset: subset of rows of O
        """

        if endrow is None:
            endrow = np.array([0])

        condition_number = np.zeros((len(endrow), 1))
        O_subset_row = None
        n = 0
        for r in endrow:
            O_subset = None
            for s in states:
                O_subset_state = self.get_subset_iterate_O(s, r, noise_removed=True)
                if O_subset is not None:
                    O_subset = np.vstack((O_subset, O_subset_state))
                else:
                    O_subset = O_subset_state

            if O_subset_row is not None:
                O_subset_row = np.vstack((O_subset_row, O_subset))
            else:
                O_subset_row = O_subset

            O_subset_row = np.unique(O_subset_row, axis=0)

            # Calculate condition # as ratio of minimum & maximum singular values
            U, E, V = np.linalg.svd(O_subset_row.T @ O_subset_row)
            min_sv = np.min(E)
            max_sv = np.max(E)
            cond_num = max_sv / min_sv
            condition_number[n] = cond_num
            print(r, cond_num)

            n = n + 1

        return O_subset_row, condition_number

    def get_observability_metrics(self, state, rows=None):
        """ Computes observability metrics for a given state.

            Inputs
                state:      state index of interest (start at 1)
                rows:       row indices to check
            Outputs
                O_subset: subset of rows of O needed for the state of interest to be observable
                rows_of_O_other_needed_for_observability: rows of O that have a nonzero entry needed
                for the state of interest to be observable
        """

        # Find the subset of O required to estimate the state of interest
        O_subset, rows_of_O_other_needed_for_observability, O_subset_augmented, observable \
            = self.minimum_subset_O(state, rows=rows)

        # Calculate SVD & condition #
        condition_number = []
        min_singular_value = []
        for r in range(len(O_subset)):
            U, E, V = scipy.linalg.svd(O_subset[r])

            # Calculate condition # as ratio of minimum & maximum singular values
            min_sv = np.min(E)
            max_sv = np.max(E)
            cond_num = max_sv / min_sv

            # Store metrics
            condition_number.append(cond_num)
            min_singular_value.append(min_sv)

        # Compile in dicts for output
        metric = {'condition_number': condition_number,
                  'min_singular_value': min_singular_value}

        state_data = {'O_subset': O_subset,
                      'rows_of_O_other_needed_for_observability': rows_of_O_other_needed_for_observability,
                      'observable': observable}

        return metric, state_data


class ObservabilityMatrix:
    def __init__(self, system, t, x, u):
        """ Construct a sliding observability matrix for a state trajectory.

            Inputs
                system:             system ode time step function
                                    takes initial condition vector, time vector, & inputs as columns in array
                                    system_ode.simulate(x0, time, input)
                t:                  time vector
                x:                  state trajectory, where columns of array are the states
                u:                  inputs, where columns are each input
        """
        self.system = system  # system object, used to simulate measurements
        self.t = t  # nominal time vector
        self.x = x  # nominal state trajectory
        self.u = u  # nominal input used to generate state trajectory
        self.dt = np.mean(np.diff(self.t))  # sampling time
        self.N = len(self.t)  # # of data points

        # To store data
        self.w = []
        self.O_all = []
        self.deltay_all = []
        self.O_time = []

    def sliding_O(self, time_resolution=None, simulation_time=None, eps=0.001):
        """ Calculate the observability matrix O for every point along a nominal state trajectory.

            Inputs
                simulation_time:    simulation time for each calculation of O
                time_resolution:    how often to calculate O along the nominal state trajectory.
                eps:                amount to perturb initial state
                measurement_type:   how to compute the measurement of the simulated system
                                    'linear', 'divide_first_two_states', or 'multiply_first_two_states'

            Outputs
                allO:               a list of the O's calculated at set times along the nominal state trajectory
                O_time:             the times O was computed
        """

        # Set time_resolution to every point, if not specified
        if time_resolution is None:
            time_resolution = self.dt

        # If time_resolution is a vector, then use the entries as the indices to calculate O
        if not np.isscalar(time_resolution) or (time_resolution == 0):  # time_resolution is a vector
            O_index = time_resolution
        else:  # evenly space the indices to calculate O
            # Resolution on nominal trajectory to calculate O, measured in indices
            time_resolution_index = np.round(time_resolution / self.dt, decimals=0).astype(int)

            # All the indices to calculate O
            O_index = np.arange(0, self.N, time_resolution_index)  # indices to compute O

        O_time = O_index * self.dt  # times to compute O
        n_point = len(O_index)  # # of times to calculate O

        # Set simulation_time to fill space between time_resolution
        if simulation_time is None:
            simulation_time = time_resolution

        # The size of the simulation time
        simulation_index = 1 + np.round(simulation_time / self.dt, decimals=0).astype(int)

        # Calculate O for each point on nominal trajectory
        O_all = []  # where to store the O's
        deltay_all = []  # where to store the O's
        for n in range(n_point):  # each point on trajectory
            # Start simulation at point along nominal trajectory
            x0 = np.squeeze(self.x[O_index[n], :])  # get state on trajectory & set it as the initial condition

            # Get the range to pull out time & input data for simulation
            win = np.arange(O_index[n], O_index[n] + simulation_index, 1)  # index range
            twin = self.t[win]  # time in window
            twin = twin - twin[0]  # start at 0
            uwin = self.u[win, :]  # inputs in window

            # Calculate O for window
            O, deltay = self.calculate_O(x0, twin, uwin, eps=eps)

            # print(twin.shape)
            # print(O.shape)
            O_all.append(O)  # store in list
            deltay_all.append(deltay)  # store in list

        # Store data in object
        self.O_all = O_all
        self.deltay_all = deltay_all
        self.O_time = O_time

        return O_all, O_time, deltay_all

    def calculate_O(self, x0, twin, uwin, eps=0.001):
        """ Numerically calculates the observability matrix O for a given system & input.

            Inputs
                x0:                 initial state
                twin:               simulation time
                twin:               simulation inputs
                eps:                amount to perturb initial state
                measurement_type:   how to compute the measurement of the simulated system
                                    'linear', 'divide_first_two_states', or 'multiply_first_two_states'

            Ouputs
                O:                  numerically calculated observability matrix
                deltay:             the difference in perturbed measurements at each time step
                                    (basically O stored in a 3D array)
        """

        # Calculate O
        dt = np.mean(np.diff(twin))
        self.w = len(twin)  # of points in time window
        delta = eps * np.eye(self.system.n)  # perturbation amount for each state
        deltay = np.zeros((self.system.p, self.system.n, self.w))  # preallocate deltay
        for k in range(self.system.n):  # each state
            # Perturb initial condition in both directions
            x0plus = x0 + delta[:, k]
            x0minus = x0 - delta[:, k]

            # Simulate measurements from perturbed initial conditions
            _, yplus = self.system.simulate(x0plus, twin, uwin)
            _, yminus = self.system.simulate(x0minus, twin, uwin)

            # Calculate the numerical Jacobian
            deltay[:, k, :] = np.array(yplus - yminus).T

        # Construct O by stacking the 3rd dimension of deltay along the 1st dimension, O is a (p*m x n) matrix
        O = np.zeros((1, self.system.n))  # make place holder 1st row
        for k in range(deltay.shape[2]):  # along 3rd dimension
            O = np.vstack((O, deltay[:, :, k]))  # stack

        O = O[1:, :]  # remove 1st row
        O = (dt ** 0.5) * O / (2 * eps)  # normalize by 2 times the perturbation amount & by the sampling time

        return O, deltay

    def calculate_O_continuous(self, x0, twin, uwin, eps=0.001,
                               measurement_type='linear', system_type='continuous', n_derivative=1):
        """ Numerically calculates the observability matrix O for a given system & input.

            Inputs
                x0:                 initial state
                twin:               simulation time
                twin:               simulation inputs
                eps:                amount to perturb initial state
                measurement_type:   how to compute the measurement of the simulated system
                                    'linear', 'divide_first_two_states', or 'multiply_first_two_states'

            Ouputs
                O:                  numerically calculated observability matrix
                deltay:             the difference in perturbed measurements at each time step
                                    (basically O stored in a 3D array)
        """

        # Calculate O
        self.w = len(twin)  # of points in time window
        delta = eps * np.eye(self.system.n)  # perturbation amount for each state
        deltay = np.zeros((self.system.p, self.system.n, self.w))  # preallocate deltay
        for k in range(self.system.n):  # each state
            # Perturb initial condition in both directions
            x0plus = x0 + delta[:, k]
            x0minus = x0 - delta[:, k]

            # Simulate measurements from perturbed initial conditions
            _, yplus = self.system.simulate(x0plus, twin, uwin, measurement_type, system_type)
            _, yminus = self.system.simulate(x0minus, twin, uwin, measurement_type, system_type)

            if system_type == 'continuous':
                # Derivatives instead
                yplus = get_column_derivatives(yplus, n_derivative=n_derivative)
                yminus = get_column_derivatives(yminus, n_derivative=n_derivative)

            # Calculate the numerical Jacobian
            deltay[:, k, :] = np.array(yplus - yminus).T

        # Construct O by stacking the 3rd dimension of deltay along the 1st dimension, O is a (p*m x n) matrix
        O = np.zeros((1, self.system.n))  # make place holder 1st row
        for k in range(deltay.shape[2]):  # along 3rd dimension
            O = np.vstack((O, deltay[:, :, k]))  # stack
        O = O[1:, :]  # remove 1st row
        O = O / (2 * eps)

        return O, deltay


def get_column_derivatives(Y, n_derivative=1, dt=1):
    """ Calculate numerical derivatives of columns in (m x n) matrix Y = [y1, y2, ..., yn]
        where yn is a (m x 1) vector.

        Return stacked Y & derivatives:
            [y1, y2, ... yn, y1_dot, y2_dot, ... yn_dot, y1_dotdot, y2_dotdot, ... y2_dotdot, ...]

        Inputs
            Y:              matrix where we take the derivative of the coloumns
            n_derivative:   # of derivatives to take
            dt:             sampling time

        Outputs
            dYdt:           stacked derivatives
    """

    # If no derivatives are requested, then return the input matrix directly
    if n_derivative == 0:
        return Y

    # Ensure Y is 2D & a column vector
    if len(Y.shape) == 1:
        Y = np.atleast_2d(Y)  # make 2D
        Y = Y.T  # column vector

    n_point = Y.shape[0]  # # of points in time (length of columns)
    n_column = Y.shape[1]  # each column is separate vector to take derivative of

    # Take derivatives of each column & append to output
    dYdt = []
    for c in range(n_column):  # each columns
        y_derivatives = np.zeros((n_point, n_derivative + 1))  # preallocate space for all derivatives of current column
        y_derivatives[:, 0] = Y[:, c]  # store column
        for k in range(n_derivative):  # each derivative
            # Take derivative of prior column & store
            y = y_derivatives[:, k]
            _, dydt = pynumdiff.finite_difference.second_order(y, dt)
            y_derivatives[:, k + 1] = dydt
            print(k)

        dYdt.append(y_derivatives)  # store all derivatives of each column in list

    dYdt = np.hstack(dYdt)  # concatenate into one matrix

    return dYdt


def analytical_observability_gramian(A, C, system_type, n_derivatives=1000):
    """ Calculate the analytical observability gramian for a given system A & C matrices.

        Inputs
            A:              system transition matrix (n x n)
            C:              system measurment/output matrix (n x p)
            system_type:    'continuous' or 'discrete'
            n_derivatives:  # of derivatives to use when calcualting discrete observability gramian,
                            has no effect for continuous time system

        Outputs
            Wo:             observability gramian
    """

    # Calculate observability gramian based on system type
    if system_type == 'continuous':
        # Solve the Lyapanuv equation
        Wo = -scipy.linalg.solve_continuous_lyapunov(A.T, C.T @ C)
    elif system_type == 'discrete':
        # Discrete summation
        Wo = np.zeros_like(A)
        for t in range(0, n_derivatives):
            Wo += np.linalg.matrix_power(A.T, t) @ C.T @ C @ np.linalg.matrix_power(A, t)
    else:
        raise Exception('"system_type" must be "continuous" or "discrete"')

    return Wo


def get_matrix_rows_iterative(A, stepsize=1, subfunction=None):
    """ For a (n x m) matrix, iteratively pulls out the collection of 1:k rows,
        where k increases in increments of stepsize. Returns the growing collection as a list Ai.

        Output list size follows this trend:
            Ai[0] = (1*stepsize x m),
            Ai[1] = (2*stepsizex m)
                        .
                        .
                        .
            Ai[-1] = (n x m) = A

        Inputs
            A:              matrix of size (n x m)
            stepsize:       # of rows to increment in each iteration, n/stepsize must be an integer value
            subfunction:    function to apply to each iteration of rows, out put

        Outputs
            Ai:             list of iteratively gorwing sub-matrices
    """

    n = A.shape[0]  # number of row
    m = A.shape[1]

    n_steps = n / stepsize  # number of steps required
    if n_steps != np.round(n_steps, decimals=0):  # check if n/stepsize is an integer
        raise Exception('The # of rows in A divided by the step size must be an integer')

    Ai = []  # list to store collection of iteratively gorwing sub-matrices
    suboutput = []  # list to store output of subfunction
    for k in range(m, n + stepsize, stepsize):  # increments of stepsize
        # print(k)
        Ak = A[0:k, :]  # collection of rows of A
        Ai.append(Ak)  # append to list

        # Apply a function to each collection of rows, if subfunction is given
        if subfunction is not None:
            output = subfunction(Ak)  # must take only matrix as input
        else:
            output = None

        suboutput.append(output)

    # # Duplicate last element of lists to ensure we keep the indices consistent
    # Ai.append(Ai[-1])
    # suboutput.append(suboutput[-1])

    # If subfunction outputs a dictionary, then convert list of dictionaries to dictionary of lists
    if type(suboutput[0]) is dict:
        suboutput = list_of_dicts_to_dict_of_lists(suboutput)

    return Ai, suboutput


def calculate_matrix_metrics(O, transpose=True):
    """ Calculates matrix metrics and returns in dictionary.

        Inputs
            O:              input matrix

        Outputs
            metrics:        dictionary contaning metrics
    """

    if transpose:
        W = O.T @ O
    else:
        W = O

    # Calculate SVD, minimum singular value, & condition #
    U, E, V = scipy.linalg.svd(W)

    # Calculate condition # as ratio of minimum & maximum singular values
    min_singular_value = np.min(E)
    max_singular_value = np.max(E)
    condition_number = max_singular_value / min_singular_value

    # Collect data in dictionary for output
    if W.shape[0] > 10: # don't store W if it is too big
        W = None

    metrics = {'W': W,
               'condition_number': condition_number,
               'min_singular_value': min_singular_value,
               'max_singular_value': max_singular_value,
               'E': E}

    return metrics


def list_of_dicts_to_dict_of_lists(list_of_dicts, keynames=None):
    """ Takes a list contaning dictionary with the same key names &
        converts it to a single dictionary where each key is a list.

        Inputs
            list_of_dicts:      input list
            keynames:           if None then use all the keys, otherwise set the key names here as a list of strings

        Outputs
            dict_of_lists:      output dictionary
    """

    # Get all the key names if not given as input
    if keynames is None:
        keynames = list_of_dicts[0].keys()  # use 1st dict to get key names

    # Create output dictionary
    dict_of_lists = {}
    for k in keynames:
        dict_of_lists[k] = []  # each key is a list

        # Get the values from the dictionaries & append to lists in output dictionaries
        for n in list_of_dicts:
            dict_of_lists[k].append(n[k])

    return dict_of_lists
