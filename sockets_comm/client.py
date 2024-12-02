import socket
import pickle
import numpy as np

def send_message(message_dict, host='127.0.0.1', port=65432):
    try:
        with socket.create_connection((host, port)) as client_socket:
            # Serialize the message dictionary
            data = pickle.dumps(message_dict)
            client_socket.sendall(data)
            
            # Receive the response
            response = client_socket.recv(4096)
            response_data = pickle.loads(response)
            print("Response from server:")
            print(response_data)
    except Exception as e:
        print(f"Error communicating with server: {e}")

if __name__ == "__main__":
    # Example NumPy array
    array = np.array([[1, 2, 3], [4, 5, 6]])

    # Send a message with a NumPy array to be processed
    message_with_pcd = {"command": "process_pcd", "pcd": array}
    send_message(message_with_pcd)

    # Send a control message to check server status
    control_message = {"command": "status", "pcd": None}
    send_message(control_message)

    # Send a disconnect message
    disconnect_message = {"command": "disconnect", "pcd": None}
    send_message(disconnect_message)
