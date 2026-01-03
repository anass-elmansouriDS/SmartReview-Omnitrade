def str2bool(v):
    """
    Function for turning string values to Boolean values.

    @Param v (str): The string value to turn into a Boolean value.

    @Return: The Boolean value.
    """
    v = v.lower()
    if v in ("true", "t"):
        return True
    if v in ("false", "f"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v}")