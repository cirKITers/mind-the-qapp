from typing import Optional

def data_is_valid(main_data: Optional[dict], page_data: Optional[dict]) -> bool:
    """
    Checks if main data and page data exist and contain non-None values

    :param main_data: dict, optional: Main data from sidebar
    :param page_data: dict, optional: Page specific data
    :return: bool: Whether data is valid
    """
    return not (page_data is None or main_data is None or \
            any([ p is None for p in page_data.values()] + \
                [ p is None for p in main_data.values() ]))
