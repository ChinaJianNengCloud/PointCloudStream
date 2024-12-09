from zeroconf import Zeroconf, ServiceBrowser
import socket
import pickle
import numpy as np

# Configuration dictionary
CONFIG = {
    "service_type": "_agent._tcp.local.",
    "discovery_timeout": 5,
}

class ServiceListener:
    def __init__(self):
        self.server_address = None

    def remove_service(self, zeroconf, type, name):
        print(f"Service {name} removed")

    def add_service(self, zeroconf, type, name):
        info = zeroconf.get_service_info(type, name)
        if info:
            print(f"Service {name} added, server IP: {socket.inet_ntoa(info.addresses[0])}")
            self.server_address = (socket.inet_ntoa(info.addresses[0]), info.port)

def discover_server(config):
    zeroconf = Zeroconf()
    listener = ServiceListener()
    browser = ServiceBrowser(zeroconf, config["service_type"], listener)

    try:
        print("Searching for services...")
        import time
        time.sleep(config["discovery_timeout"])  # Wait for responses
    finally:
        zeroconf.close()

    return listener.server_address

def send_message(message_dict, server_address):
    try:
        with socket.create_connection(server_address) as client_socket:
            data = pickle.dumps(message_dict)
            client_socket.sendall(len(data).to_bytes(4, byteorder='big') + data)

            response_length = int.from_bytes(client_socket.recv(4), byteorder='big')
            response_data = b""
            while len(response_data) < response_length:
                packet = client_socket.recv(4096)
                if not packet:
                    break
                response_data += packet

            response = pickle.loads(response_data)
            print("Response from server:")
            print(response)
    except Exception as e:
        print(f"Error communicating with server: {e}")

if __name__ == "__main__":
    # Discover server
    server_address = discover_server(CONFIG)
    if not server_address:
        print("No server found!")
    else:
        print(f"Server discovered at {server_address}")

        # Example NumPy array
        array = np.random.rand(300000, 6)

        # Send a message with a NumPy array to be processed
        message_with_pcd = {"command": "process_pcd", "pcd": array}
        send_message(message_with_pcd, server_address)
