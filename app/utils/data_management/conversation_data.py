class ConversationData:
    def __init__(self):
        self.conversation = []  # Store conversation as a list of tuples (role, message)

    def reset(self):
        self.conversation = []

    def append(self, role: str, message: str):
        """
        Appends a message to the conversation.
        
        Args:
            role (str): Role of the sender, either "User" or "Agent".
            message (str): The message content.
        """
        if role not in ["User", "Agent"]:
            raise ValueError("Role must be 'User' or 'Agent'.")
        self.conversation.append((role, message))

    def get_terminal_conversation(self) -> str:
        """
        Formats the conversation for terminal display with ANSI colors 
        and role prefixes.
        
        Returns:
            str: The formatted conversation string for terminal display.
        """
        formatted_conversation = "\n" + "-" * 50 + "\n"
        for role, message in self.conversation:
            if role == "User":
                # Green for user
                formatted_conversation += f"\033[92mUser: {message}\033[0m\n"
            elif role == "Agent":
                # Blue for agent
                formatted_conversation += f"\033[94mAgent: {message}\033[0m\n"
            formatted_conversation += "-" * 50 + "\n"  # Add separator line
        return formatted_conversation

    def get_qt_format_conversation(self) -> str:
        """
        Formats the conversation for Qt text editor display using HTML
        with color styling.
        
        Returns:
            str: The formatted conversation string for Qt text editor.
        """
        formatted_conversation = "<html><body>"
        for role, message in self.conversation:
            if role == "User":
                # Green for user
                formatted_conversation += f"<p><span style='color:green;'><b>User:</b> {message}</span></p>"
            elif role == "Agent":
                # Blue for agent
                formatted_conversation += f"<p><span style='color:blue;'><b>Agent:</b> {message}</span></p>"
            formatted_conversation += "<hr>"  # Add horizontal line as separator
        formatted_conversation += "</body></html>"
        return formatted_conversation


if __name__ == "__main__":
    # Create an instance of ConversationData
    conversation = ConversationData()

    # Append messages to the conversation
    conversation.append("User", "Hello, how are you?")
    conversation.append("Agent", "I'm good, thank you! How can I assist you today?")
    conversation.append("User", "I need help with a Python project.")
    conversation.append("Agent", "Sure, tell me more about your project.")

    # Show the conversation in the terminal
    print(conversation.get_terminal_conversation())

    # Get the conversation in Qt format
    qt_formatted_conversation = conversation.get_qt_format_conversation()
    print(qt_formatted_conversation)  # This is HTML and suitable for use in Qt text editors
