import json
import os
import traceback
from typing import Any, Dict, Optional
import streamlit.components.v1 as components

# Create a _RELEASE constant. We'll set this to False while we're developing
# the component, and True when we're ready to package and distribute it.
# (This is, of course, optional - there are innumerable ways to manage your
# release process.)
_RELEASE = os.environ.get("FLOWCO_DEV", "0") == "0"

# Declare a Streamlit component. `declare_component` returns a function
# that is used to create instances of the component. We're naming this
# function "_component_func", with an underscore prefix, because we don't want
# to expose it directly to users. Instead, we will create a custom wrapper
# function, below, that will serve as our component's public API.

# It's worth noting that this call to `declare_component` is the
# *only thing* you need to do to create the binding between Streamlit and
# your component frontend. Everything else we do in this file is simply a
# best practice.

if _RELEASE:
    pass
    # print("RELEASE MODE")
else:
    print("DEVELOPMENT MODE")

if not _RELEASE:
    _component_func = components.declare_component(
        # We give the component a simple, descriptive name ("maxgraph_component"
        # does not fit this bill, so please choose something better for your
        # own component :)
        "mxgraph_component",
        # Pass `url` here to tell Streamlit that the component will be served
        # by the local dev server that you run via `npm run start`.
        # (This is useful while your component is in development.)
        url="http://localhost:3000",
    )
else:
    # When we're distributing a production version of the component, we'll
    # replace the `url` param with `path`, and point it to the component's
    # build directory:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    _component_func = components.declare_component("mxgraph_component", path=build_dir)


_counter = 0


# Create a wrapper function for the component. This is an optional
# best practice - we could simply expose the component function returned by
# `declare_component` and call it done. The wrapper allows us to customize
# our component's API: we can pre-process its input args, post-process its
# output value, and add a docstring for users.
def mxgraph_component(
    key: str,
    diagram: Dict[str, Any],
    editable: bool,
    selected_node: Optional[str] = None,
    zoom: Optional[str] = None,
    refresh_phase: int = 0,
    dummy: Optional[str] = None,
    clear=False,
) -> int:
    """Create a new instance of "maxgraph_component".

    Parameters
    ----------
    key: str or None
        An optional key that uniquely identifies this component. If this is
        None, and the component's arguments are changed, the component will
        be re-mounted in the Streamlit frontend and lose its current state.


    """

    # print("mxgraph_component called")
    # traceback.print_stack()

    # Call through to our private component function. Arguments we pass here
    # will be sent to the frontend, where they'll be available in an "args"
    # dictionary.
    #
    # "default" is a special argument that specifies the initial return
    # value of the component before the user has interacted with it.
    component_value = _component_func(
        key=key,
        diagram=diagram,
        editable=editable,
        selected_node=selected_node,
        refresh_phase=refresh_phase,
        zoom=zoom,
        default={
            "command": "update",
            "diagram": json.dumps(diagram),
            "selected_node": selected_node,
            "height": 600,  # just a default value
        },
        forced=dummy != None,
        clear=clear,
        dummy=dummy,  # pass in to force update when dummy changes
    )

    # We could modify the value returned from the component if we wanted.
    # There's no need to do this in our simple example - but it's an option.
    return component_value
