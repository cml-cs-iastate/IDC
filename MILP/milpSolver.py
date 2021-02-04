import collections

from mip import Model, xsum, minimize, BINARY, OptimizationStatus, INTEGER


# Tested
def milp_solver(data, vocabulary, num_classes, vocab_size):
    """
    Mixed-Integer Linear Programing with mip
    :param data: matrix of one-hot representation of each document per class       {num_classes: [None, vocab_size]}
    :param vocabulary: the super vector of the unique words
    :param num_classes: the number of classes
    :param vocab_size: the vocabulary size in terms number of words
    """

    classes = set(i for i in range(num_classes))

    print('Start the milp function!')

    # ------------------------------------------------------------------------------------------------------------------
    # Create the optimization model
    model = Model()
    print('Optimization model is created!')

    # ------------------------------------------------------------------------------------------------------------------
    # Define the variables
    # MILP Variable: (to optimize)  shape [num_classes, vocab_size]
    V = [[model.add_var(var_type=BINARY) for _ in range(vocab_size)] for _ in range(num_classes)]
    print('Optimization variables are created!')

    # ------------------------------------------------------------------------------------------------------------------
    # Optimization function
    # xsum(V[c][j] for c in range(num_classes) for j in range(vocab_size)) +
    model.objective = minimize(xsum(d[j] * V[c_prime][j]
                                    for c in classes
                                    for d in data[c]
                                    for c_prime in classes - {c}
                                    for j in range(vocab_size)))

    print('Optimization objective function is done!')

    # ------------------------------------------------------------------------------------------------------------------
    # Optimization functions (Constraints)
    # Equation (1)
    """
    Each interpretation feature must belong to only one category (Tested)
    """
    for j in range(vocab_size):
        model.add_constr(xsum(V[c][j] for c in range(num_classes)) <= 1)
    print('Optimization subjective equation (1) is done!')

    # Equation (2)
    """
    Each document of class c must have at least one interpretation feature that belong to class c (Tested)
    """
    for c in range(num_classes):
        for d in data[c]:
            model.add_constr(xsum(d[j] * V[c][j] for j in range(vocab_size)) >= 1)
    print('Optimization subjective equation (2) is done!')

    # ------------------------------------------------------------------------------------------------------------------
    # run optimization
    status = model.optimize()
    print('Optimization is over!')

    # ------------------------------------------------------------------------------------------------------------------
    # Display the solution
    if status == OptimizationStatus.OPTIMAL:
        print('OPTIMAL solution has been found!')
        print('optimal solution cost {} found'.format(model.objective_value))
    elif status == OptimizationStatus.FEASIBLE:
        print('FEASIBLE solution has been found!')
        print('sol.cost {} found, best possible: {}'.format(model.objective_value, model))
    else:
        print('No OPTIMAL nor FEASIBLE solution has been found!')
        print('no feasible solution found, lower bound is: {}'.format(model.objective_bound))

    # ------------------------------------------------------------------------------------------------------------------
    # Post process solution
    milp_if = collections.defaultdict(set)
    if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
        for i in range(num_classes):
            for j in range(vocab_size):
                if V[i][j].x > 0:
                    milp_if[i].add(vocabulary[j])
    else:
        print('The problem does not have an optimal solution.')

    print('milp function is over!')

    return milp_if
