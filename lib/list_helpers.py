def pad_insert(list_in: list, item, index: int):
    """insert an item into list at the specified index, extending the list
    with zeroes first as needed.
    
    Parameters:
      list_in (list): the list to extend
      item: the item to insert
      index (int): the index at which to insert

    Output:
      `list_in` with `item` inserted
    """
    if index + 1 < len(list_in):
        list_in.extend([0] * (index + 1 - len(list_in)))
    return list_in

