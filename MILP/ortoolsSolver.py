import collections

from ortools.linear_solver import pywraplp

'''https://developers.google.com/optimization/mip/integer_opt'''


# Tested
def milp_solver(data, vocabulary, num_classes, vocab_size):
    """
    Mixed-Integer Linear Programing with or_tools
    :param data: matrix of one-hot representation of each document per class        {num_classes: [None, vocab_size]}
    :param vocabulary: the super vector of the unique words                         [vocab_size]
    :param num_classes: the number of classes                                       int
    :param vocab_size: the vocabulary size in terms number of words                 int
    """

    classes = set(i for i in range(num_classes))
    print(f'classes : {classes}')

    print('Start the milp function!')

    # ------------------------------------------------------------------------------------------------------------------
    # Create the optimization model with the CBC backend.
    # model = pywraplp.Solver.CreateSolver('simple_mip_program', solver_id='SCIP')
    model = pywraplp.Solver.CreateSolver(solver_id='SCIP')
    print('Optimization model is created!')

    # ------------------------------------------------------------------------------------------------------------------
    # Define the variables
    # MILP Variable: (to optimize)  shape [num_classes, vocab_size]
    V = collections.defaultdict(dict)
    for i in range(num_classes):
        for j in range(vocab_size):
            V[i][j] = model.BoolVar('%s%s' % (i, j))

    print('Optimization variables are created!')
    print(f'Number of variables = {model.NumVariables()}')

    # ------------------------------------------------------------------------------------------------------------------
    # Optimization functions (Constraints)
    # Equation (1)
    """
    Each interpretation feature is assigned to at most one category (Tested)
    """
    for j in range(vocab_size):
        model.Add(model.Sum([V[c][j] for c in range(num_classes)]) <= 1)
    print('Optimization subjective equation (1) is created!')

    # Equation (2)
    """
    Each document of class c must have at least one interpretation feature that belong to class c (Tested)
    """
    for c in range(num_classes):
        for d in data[c]:
            model.Add(model.Sum([d[j] * V[c][j] for j in range(vocab_size)]) >= 1)
    print('Optimization subjective equation (2) is created!')

    print('Number of constraints =', model.NumConstraints())

    # ------------------------------------------------------------------------------------------------------------------
    # Optimization (Objective) function
    # sum(V[c][j] for c in range(num_classes) for j in range(vocab_size)) +
    objective_terms = []
    for c in classes:
        for d in data[c]:
            for c_prime in classes - {c}:
                for j in range(vocab_size):
                    objective_terms.append(d[j] * V[c_prime][j])
    model.Minimize(model.Sum(objective_terms))

    print('Optimization objective function is created!')

    # ------------------------------------------------------------------------------------------------------------------
    # run optimization
    print('Optimization started!')
    status = model.Solve()
    print('Optimization is over!')

    # ------------------------------------------------------------------------------------------------------------------
    # Display and post process the solution
    milp_if = collections.defaultdict(set)
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        print('Solution:')
        print(f'Objective value = {model.Objective().Value()} \n')
        for i in range(num_classes):
            for j in range(vocab_size):
                # print(f'Variable {j} of class {i} = {V[i][j].solution_value()}')
                if V[i][j].solution_value() > 0:
                    milp_if[i].add(vocabulary[j])
    else:
        print('The problem does not have an optimal or feasible solution.')

    print('\nAdvanced usage:')
    print('Problem solved in %f milliseconds' % model.wall_time())
    print('Problem solved in %d iterations' % model.iterations())
    print('Problem solved in %d branch-and-bound nodes' % model.nodes())

    print('milp function is over!')

    return milp_if
