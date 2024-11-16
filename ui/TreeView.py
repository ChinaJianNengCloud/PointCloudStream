import open3d.visualization.gui as gui


class ModifiedTreeView:
    def __init__(self):
        self.tree = gui.TreeView()
        self.item_to_parent = {}  # Mapping of item IDs to their parent item IDs
        self.item_to_text = {}    # Mapping of item IDs to their corresponding text
        self.item_to_level = {}   # Mapping of item IDs to their levels
        self.item_to_root_text = {}  # Mapping of item IDs to their root parent's text
        self._on_selection_callback = None  # User-defined callback for selection changes
        self.selected_item = self.SelectedItem()
        # Enable selection of items with children
        self.tree.can_select_items_with_children = True

        # Set the selection change callback
        self.tree.set_on_selection_changed(self._on_tree_selection)

    def add_item(self, parent_id, text, level, root_text=None):
        """
        Adds an item to the TreeView and tracks its metadata.

        :param parent_id: The parent item ID.
        :param text: The text for the new item.
        :param level: The hierarchical level of the new item.
        :param root_text: The root parent's text for the new item (defaults to its own text if root level).
        :return: The ID of the new item.
        """
        item_id = self.tree.add_text_item(parent_id, text)
        self.item_to_parent[item_id] = parent_id
        self.item_to_text[item_id] = text
        self.item_to_level[item_id] = level
        self.item_to_root_text[item_id] = root_text if root_text else text
        return item_id

    def set_on_selection_changed(self, callback):
        """
        Sets a user-defined callback to be called when the selection changes.

        :param callback: A callable with the signature callback(item_id, level, index_in_level, parent_text, root_text).
        """
        self._on_selection_callback = callback

    def get_tree_widget(self):
        """
        Returns the underlying TreeView widget.

        :return: The TreeView widget.
        """
        return self.tree

    def _on_tree_selection(self, new_item_id):
        """
        Called when a TreeView item is selected. Calls the user-defined callback if provided.
        """
        # Get metadata for the selected item
        level = self.item_to_level.get(new_item_id, -1)
        parent_id = self.item_to_parent.get(new_item_id, None)
        parent_text = self.item_to_text.get(parent_id, "None")
        root_text = self.item_to_root_text.get(new_item_id, "None")

        # Find all siblings (items with the same parent)
        siblings = [
            item_id for item_id, pid in self.item_to_parent.items() if pid == parent_id
        ]
        index_in_level = siblings.index(new_item_id) if new_item_id in siblings else -1

        self.selected_item.set_attr(new_item_id, level, index_in_level, parent_text, root_text)
        # Call the user-defined callback, if any
        if self._on_selection_callback:
            self._on_selection_callback(self.selected_item)

    class SelectedItem:
        def __init__(self, item_id=None, level=None, index_in_level=None, parent_text=None, root_text=None):
            self.item_id = item_id
            self.level = level
            self.index_in_level = index_in_level
            self.parent_text = parent_text
            self.root_text = root_text

        def __eq__(self, value: str) -> bool:
            return self.item_id == value

        def set_attr(self, item_id, level, index_in_level, parent_text, root_text):
            self.item_id = item_id
            self.level = level
            self.index_in_level = index_in_level
            self.parent_text = parent_text
            self.root_text = root_text
        
        def reset(self):
            self.item_id = None
            self.level = None
            self.index_in_level = None
            self.parent_text = None
            self.root_text = None

# Example usage

if __name__ == "__main__":
    def custom_selection_callback(item:ModifiedTreeView.SelectedItem):
        print(
            f"Root Parent Text: {item.root_text}\n"
            f"Custom Callback -> Selected Item ID: {item.item_id}, "
            f"Level: {item.level}, Index in Level: {item.index_in_level}, "
            f"Parent Text: {item.parent_text}"
        )

    gui.Application.instance.initialize()

    # Create a window
    window = gui.Application.instance.create_window(
        "Modified TreeView Example", 1280, 720
    )

    # Initialize ModifiedTreeView
    tree_view = ModifiedTreeView()

    # Sample data
    data = {'76ec013f69144905bfd7dba0472ee7c2': {'prompt': 'prompta dd sda sds', 'pose': [[4, 5, 6]]}, 'd6ea45ccd41541a38966605f84a1fa84': {'prompt': 'prompt1 ass dsa s', 'pose': [[2, 5, 6], [2, 5, 6], [2, 2, 6], [2, 5, 1]]}, 'bf365dd20f8d4b399b2cab9ea6ec7cfb': {'prompt': 'prompt dsa s', 'pose': [[2, 5, 6]]}, '89819b12a56b4195b628ddf3f65ac882': {'prompt': 'prompt1', 'pose': [[2, 5, 6]]}}

    # Populate the TreeView
    for key, value in data.items():
        root_id = tree_view.add_item(tree_view.tree.get_root_item(), key, level=1)

        # Add 'prompt' field
        # prompt_id = tree_view.add_item(root_id, "Prompt", level=2, root_text=key)
        tree_view.add_item(root_id, "Prompt: " + value["prompt"], level=2, root_text=key)

        # Add 'pose' field
        pose_id = tree_view.add_item(root_id, "Pose", level=2, root_text=key)
        for i, pose in enumerate(value["pose"]):
            pose_text = f"Pose {i + 1}: [{', '.join(f'{v:.2f}' for v in pose)}]"
            tree_view.add_item(pose_id, pose_text, level=3, root_text=key)

    # Set the custom selection callback
    tree_view.set_on_selection_changed(custom_selection_callback)

    # Add the TreeView widget to the window
    window.add_child(tree_view.get_tree_widget())

    # Run the application
    gui.Application.instance.run()
