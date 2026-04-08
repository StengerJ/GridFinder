from importlib.resources import files


def image_path(name: str) -> str:
    return str(files("gridworld.modules").joinpath("images", name))
