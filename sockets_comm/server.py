import socket
import pickle
import numpy as np
import threading

def process_data(data):
    try:
        # Deserialize the received data
        msg_dict:dict = pickle.loads(data)
        command = msg_dict.get('command', None)
        pcd = msg_dict.get('pcd', None)

        # Handle different commands
        if command == "process_pcd" and isinstance(pcd, np.ndarray):
            print("Processing NumPy array:")
            print(pcd)
            result = pcd * 2
            return pickle.dumps({"status": "success", "result": result})
        elif command == "status":
            return pickle.dumps({"status": "success", "message": "Server is running."})
        elif command == "disconnect":
            print("Client requested disconnect.")
            return pickle.dumps({"status": "success", "message": "Disconnected."})
        else:
            return pickle.dumps({"status": "error", "message": "Unknown command."})
    except Exception as e:
        print(f"Error processing data: {e}")
        return pickle.dumps({"status": "error", "message": str(e)})

def handle_client(conn, addr):
    print(f"Connected by {addr}")
    try:
        while True:
            data = conn.recv(4096)
            if not data:
                break
            response = process_data(data)
            conn.sendall(response)
    except Exception as e:
        print(f"Connection error with {addr}: {e}")
    finally:
        conn.close()
        print(f"Connection closed with {addr}")

def start_server(host='0.0.0.0', port=65432):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen()
    print(f"Server listening on {host}:{port}")

    try:
        while True:
            conn, addr = server_socket.accept()
            client_thread = threading.Thread(target=handle_client, args=(conn, addr))
            client_thread.start()
    except KeyboardInterrupt:
        print("Server shutting down.")
    finally:
        server_socket.close()

if __name__ == "__main__":
    start_server()
