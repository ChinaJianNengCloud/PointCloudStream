from typing import List
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QLabel, QFrame, QVBoxLayout, QHBoxLayout, QWidget, QSizePolicy, QScrollArea
from PySide6.QtCore import Qt, QTimer

class ChatBubble(QFrame):
    """
    A widget to display a single chat message in a bubble format.
    """
    def __init__(self, text, em, is_user=True, parent=None):
        super().__init__(parent)
        self.em = em  # Use em as the base unit
        self.setFrameShape(QFrame.Shape.NoFrame)
        bubble_color = "#0084ff" if is_user else "#3C3F41"
        self.setStyleSheet(f"""
            QFrame {{
                border-radius: {self.em}px;
                padding: {0.1 * self.em}px;  /* Adjusted padding for proportional scaling */
                background-color: {bubble_color};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(self.em, self.em, self.em, self.em)  # Proportional margins
        layout.setSpacing(0)

        self.label = QLabel(text, self)
        self.label.setWordWrap(True)
        self.label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.label.setStyleSheet(f"""
            color: white; 
            font-size: {1.2 * self.em}px;
        """)

        layout.addWidget(self.label)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def setBubbleWidth(self, viewport_width):
        """
        Adjust the bubble width dynamically based on text length
        while enforcing a maximum width relative to the viewport width.
        """
        margin = 3 * self.em  # Adjust for padding/margin around the bubble
        max_width = viewport_width - margin  # Limit bubble width to viewport width minus margin
        min_width = 5 * self.em  # Minimum width for short messages
        text_width = self.label.fontMetrics().boundingRect(self.label.text()).width()

        # Set width to be the minimum of text width or max width, but at least min_width
        adjusted_width = min(max_width, max(min_width, text_width + margin))
        self.setFixedWidth(adjusted_width)
        self.label.setFixedWidth(adjusted_width - 2 * self.em)  # Match label width with margin

        # Update height dynamically to match the content
        self.adjustSize()


class LoadingBubble(QFrame):
    """
    A bubble that shows a loading animation (e.g., rotating dots).
    """
    def __init__(self, em, parent=None):
        super().__init__(parent)
        self.em = em
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet(f"""
            QFrame {{
                border-radius: {self.em}px;
                padding: {0.1 * self.em}px;
                background-color: #3C3F41; 
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(self.em, self.em, self.em, self.em)
        layout.setSpacing(0)

        self.label = QLabel("Thinking...", self)
        self.label.setWordWrap(True)
        self.label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.label.setStyleSheet(f"color: white; font-size: {1.0 * self.em}px;")

        layout.addWidget(self.label)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        # Setup a timer to animate the thinking states
        self.thinking_states = ["Thinking.", "Thinking..", "Thinking..."]
        self.current_state_index = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(500)  # Change state every half second

    def animate(self):
        self.current_state_index = (self.current_state_index + 1) % len(self.thinking_states)
        self.label.setText(self.thinking_states[self.current_state_index])

    def setBubbleWidth(self, viewport_width):
        """
        For consistency with ChatBubble, though loading text is simple.
        """
        margin = 3 * self.em
        max_width = viewport_width - margin
        min_width = 5 * self.em
        # The width is short and simple, so just pick a reasonable width
        adjusted_width = min(max_width, max(min_width, 10 * self.em))
        self.setFixedWidth(adjusted_width)
        self.label.setFixedWidth(adjusted_width - 2 * self.em)
        self.adjustSize()


class ChatHistoryWidget(QWidget):
    """
    A widget to contain and display a vertical stack of messages.
    On resize, it adjusts each message bubble's width according to the scroll area viewport.
    This version includes a loading bubble when a user message is added,
    and removes the loading bubble before adding a non-user message.
    """
    def __init__(self, scroll_area, em, parent=None):
        super().__init__(parent)
        self.scroll_area:QScrollArea = scroll_area
        self.em = em
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(int(0.5*self.em), int(0.5*self.em), int(0.5*self.em), int(0.5*self.em))
        self.main_layout.setSpacing(int(self.em))
        self.main_layout.addStretch()

        self.bubbles:List[ChatBubble] = []
        self.loading_bubble = None

    def add_message(self, text, is_user=True):
        """
        Add a message bubble. If it's a user message, also show a loading bubble.
        If it's a non-user message, remove the loading bubble first.
        """
        if not is_user:
            # Non-user message: first remove loading bubble if exists
            self.hide_loading_bubble()

        h_layout = QHBoxLayout()
        bubble = ChatBubble(text, self.em, is_user)

        if is_user:
            # User messages: push bubble to the right
            h_layout.addStretch()
            h_layout.addWidget(bubble)
        else:
            # Agent messages: push bubble to the left
            h_layout.addWidget(bubble)
            h_layout.addStretch()

        self.main_layout.insertLayout(self.main_layout.count() - 1, h_layout)
        self.bubbles.append(bubble)
        self.adjust_bubbles_width()

        if is_user:
            # Show loading bubble after user message
            self.show_loading_bubble()

    def show_loading_bubble(self):
        # If there's already a loading bubble, do nothing
        if self.loading_bubble is not None:
            return

        h_layout = QHBoxLayout()
        self.loading_bubble = LoadingBubble(self.em)
        # Loading bubble: align like a non-user message (on the left)
        h_layout.addWidget(self.loading_bubble)
        h_layout.addStretch()

        self.main_layout.insertLayout(self.main_layout.count() - 1, h_layout)
        self.adjust_bubbles_width()

    def hide_loading_bubble(self):
        # Remove and delete the loading bubble if it exists
        if self.loading_bubble is not None:
            # Find and remove from layout
            for i in range(self.main_layout.count()):
                item = self.main_layout.itemAt(i)
                if isinstance(item, QHBoxLayout):
                    # Check this layout's items
                    for j in range(item.count()):
                        w = item.itemAt(j)
                        if w and w.widget() == self.loading_bubble:
                            # Found the loading bubble
                            # Remove it from layout
                            item.removeWidget(self.loading_bubble)
                            self.loading_bubble.deleteLater()
                            self.loading_bubble = None
                            # If h_layout is empty now, remove it
                            if item.count() == 0:
                                # Actually, since we had stretches, it won't be empty
                                # but let's just break after removing the bubble.
                                pass
                            break
                    # If we removed the bubble, break outer loop as well
                    if self.loading_bubble is None:
                        break

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.adjust_bubbles_width()

    def adjust_bubbles_width(self):
        viewport_width = self.scroll_area.viewport().width()
        if viewport_width <= 0:
            return
        for bubble in self.bubbles:
            bubble.setBubbleWidth(viewport_width)
        if self.loading_bubble is not None:
            self.loading_bubble.setBubbleWidth(viewport_width)
