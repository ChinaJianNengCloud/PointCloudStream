from PyQt5.QtWidgets import QTreeWidget, QTreeWidgetItem
from typing import Callable, Dict, Optional, List

class DataTreeWidget(QTreeWidget):
    def __init__(self) -> None:
        """
        Initializes the DataTreeWidget.

        The DataTreeWidget is a QTreeWidget that tracks the hierarchical level of each item
        and provides a way to get the root parent's text for each item.
        """
        super().__init__()
        self.item_to_level: Dict[int, int] = {}   # Mapping of item IDs to their levels
        self.item_to_root_text: Dict[int, str] = {}  # Mapping of item IDs to their root parent's text
        self._on_selection_callback: Optional[Callable[[QTreeWidgetItem, int, int, str, str], None]] = None
        self.selected_item = self.SelectedItem()
        self.itemClicked.connect(self._on_item_clicked)

    def add_item(
        self,
        parent_item: Optional[QTreeWidgetItem],
        text: str,
        level: int,
        root_text: Optional[str] = None,
    ) -> QTreeWidgetItem:
        """
        Adds an item to the QTreeWidget and tracks its metadata.

        :param parent_item: The parent QTreeWidgetItem.
        :param text: The text for the new item.
        :param level: The hierarchical level of the new item.
        :param root_text: The root parent's text for the new item (defaults to its own text if root level).
        :return: The created QTreeWidgetItem.
        """
        item = QTreeWidgetItem(parent_item, [text]) if parent_item else QTreeWidgetItem([text])
        self.addTopLevelItem(item) if not parent_item else None

        item_id = id(item)
        self.item_to_level[item_id] = level
        self.item_to_root_text[item_id] = root_text if root_text else text
        return item

    def set_on_selection_changed(self, callback: Callable[[QTreeWidgetItem, int, int, str, str], None]) -> None:
        """
        Sets a user-defined callback to be called when the selection changes.

        :param callback: A callable with the signature callback(item: QTreeWidgetItem, level: int, index_in_level: int, parent_text: str, root_text: str).
        """
        self._on_selection_callback = callback

    def _on_item_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        """
        Called when a QTreeWidgetItem is clicked. Calls the user-defined callback if provided.

        :param item: The clicked QTreeWidgetItem.
        :param column: The column index of the clicked item.
        """
        item_id: int = id(item)
        parent_item: Optional[QTreeWidgetItem] = item.parent()
        parent_id: Optional[int] = id(parent_item) if parent_item else None
        level: int = self.item_to_level.get(item_id, -1)
        parent_text: str = parent_item.text(0) if parent_item else "None"
        root_text: str = self.item_to_root_text.get(item_id, "None")

        # Find all siblings
        siblings: List[QTreeWidgetItem] = []
        if parent_item:  # If the item has a parent, get siblings from the parent
            for i in range(parent_item.childCount()):
                siblings.append(parent_item.child(i))
        else:  # If the item is a top-level item, get siblings from the invisible root
            for i in range(self.topLevelItemCount()):
                siblings.append(self.topLevelItem(i))

        index_in_level: int = siblings.index(item) if item in siblings else -1
        self.selected_item.set_attr(item, level, index_in_level, parent_text, root_text)
        # Call the user-defined callback, if any
        if self._on_selection_callback:
            self._on_selection_callback(item, level, index_in_level, parent_text, root_text)

    class SelectedItem:
        def __init__(self, item: Optional[QTreeWidgetItem] = None, level: Optional[int] = None, index_in_level: Optional[int] = None, parent_text: Optional[str] = None, root_text: Optional[str] = None) -> None:
            """
            Initializes the selected item fields.

            :param item: The selected QTreeWidgetItem.
            :param level: The hierarchical level of the selected item.
            :param index_in_level: The index of the selected item within its level.
            :param parent_text: The text of the parent item.
            :param root_text: The text of the root item.
            """
            self.item = item
            self.level = level
            self.index_in_level = index_in_level
            self.parent_text = parent_text
            self.root_text = root_text

        def __eq__(self, value: Optional[QTreeWidgetItem]) -> bool:
            """
            Compares the selected item with another QTreeWidgetItem.

            :param value: The QTreeWidgetItem to compare with.
            :return: True if the items are equal, False otherwise.
            """
            return self.item == value

        def set_attr(self, item: Optional[QTreeWidgetItem], level: Optional[int], index_in_level: Optional[int], parent_text: Optional[str], root_text: Optional[str]) -> None:
            """
            Sets the attributes of the selected item.

            :param item: The selected QTreeWidgetItem or None.
            :param level: The hierarchical level of the selected item or None.
            :param index_in_level: The index of the selected item within its level or None.
            :param parent_text: The text of the parent item or None.
            :param root_text: The text of the root item or None.
            """
            self.item = item
            self.level = level
            self.index_in_level = index_in_level
            self.parent_text = parent_text
            self.root_text = root_text
        
        def reset(self, item: Optional[QTreeWidgetItem] = None, level: Optional[int] = None, index_in_level: Optional[int] = None, parent_text: Optional[str] = None, root_text: Optional[str] = None) -> None:
            """
            Resets the selected item fields.

            :param item: The selected QTreeWidgetItem or None (default).
            :param level: The hierarchical level of the selected item or None (default).
            :param index_in_level: The index of the selected item within its level or None (default).
            :param parent_text: The text of the parent item or None (default).
            :param root_text: The text of the root item or None (default).
            """
            self.item = item
            self.level = level
            self.index_in_level = index_in_level
            self.parent_text = parent_text
            self.root_text = root_text
