import dash

dash.register_page(__name__, name="Home", path="/")

from layouts.main_page_layout import layout  # noqa
