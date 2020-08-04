class Constraint:
    """Object holding Constraint along with its user define name for easier debugging
    """

    def __init__(self, name, constraint):
        """initialize a constraint

        Args:
            name (str): name of this set of constraints
            constraint (list): a list of constraints associated with that input name for debugging
        """
        self._name = name
        self._constraint = constraint

    def get_name(self):
        """returns current constraint name
        
        Returns:
            string -- name of this set of constraints
        """
        return self._name

    def get_constraint(self):
        """returns list of constraints associated with the current constraint name
        
        Returns:
            list -- list of cvxpy.constraint
        """
        return self._constraint
