from zeroconf import Zeroconf, ServiceInfo
import socket
import pickle
import threading
import time
import os
import subprocess
import numpy as np

class ZeroconfServer:
    def __init__(self, config):
        self.host = config["host"]
        self.port = config["port"]
        self.service_name = config["service_name"]
        self.service_type = config["service_type"]
        self.service_description = config["service_description"]
        self.server_socket = None
        self.zeroconf = Zeroconf()
        self.service_info = None
        self.process_function = None  # Placeholder for external data processing function

    def set_processing_function(self, func):
        """Set the data processing function."""
        self.process_function = func

    @staticmethod
    def kill_process_using_port(port):
        """Kill the process using the specified port."""
        try:
            # Find the PID using lsof
            result = subprocess.run(
                ["lsof", "-i", f":{port}"], capture_output=True, text=True
            )
            lines = result.stdout.splitlines()

            for line in lines[1:]:
                parts = line.split()
                pid = int(parts[1])
                print(f"Killing process with PID {pid} using port {port}")
                os.kill(pid, 9)  # Forcefully terminate the process
                time.sleep(0.5)

        except Exception as e:
            print(f"Failed to kill process using port {port}: {e}")

    def check_and_free_port(self):
        """Check if a port is in use and free it if necessary."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as test_socket:
            test_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                test_socket.bind((self.host, self.port))
                print(f"Port {self.port} is free to use.")
            except OSError:
                print(f"Port {self.port} is in use. Attempting to free it.")
                self.kill_process_using_port(self.port)

    def process_data(self, data):
        """Default or external data processing function."""
        if self.process_function:
            return self.process_function(data)  # Call the external function
        else:
            return pickle.dumps({"status": "error", "message": "No processing function set."})

    @staticmethod
    def receive_full_message(conn):
        data_buffer = b""
        msg_length = int.from_bytes(conn.recv(4), byteorder='big')
        while len(data_buffer) < msg_length:
            packet = conn.recv(4096)
            if not packet:
                break
            data_buffer += packet
        return data_buffer

    def handle_client(self, conn, addr):
        print(f"Connected by {addr}")
        try:
            while True:
                data = self.receive_full_message(conn)
                if not data:
                    break
                print(f"Processing data from {addr}")
                time.sleep(3)
                response = self.process_data(data)
                print(f"Sending response to {addr}")
                conn.sendall(len(response).to_bytes(4, byteorder='big') + response)
        except Exception as e:
            print(f"Connection error with {addr}: {e}")
        finally:
            conn.close()
            print(f"Connection closed with {addr}")

    def start(self):
        """Start the server."""
        self.check_and_free_port()

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen()
        print(f"Server listening on {self.host}:{self.port}")

        # Advertise the service using Zeroconf
        self.service_info = ServiceInfo(
            self.service_type,
            f"{self.service_name}.{self.service_type}",
            addresses=[socket.inet_aton(self.host)],
            port=self.port,
            properties={"description": self.service_description},
            server=f"{self.service_name}.local.",
        )

        self.zeroconf.register_service(self.service_info)
        print("Service registered with Zeroconf.")

        try:
            while True:
                conn, addr = self.server_socket.accept()
                client_thread = threading.Thread(target=self.handle_client, args=(conn, addr))
                client_thread.start()
        except KeyboardInterrupt:
            print("Server shutting down.")
        finally:
            self.stop()

    def stop(self):
        """Stop the server and clean up resources."""
        if self.service_info:
            self.zeroconf.unregister_service(self.service_info)
        self.zeroconf.close()
        if self.server_socket:
            self.server_socket.close()
        print("Server stopped.")


if __name__ == "__main__":
    # Example usage
    def custom_processing_function(data):
        try:
            msg_dict:dict = pickle.loads(data)
            command = msg_dict.get('command', None)
            pcd = msg_dict.get('pcd', None)

            if command == "process_pcd" and isinstance(pcd, np.ndarray):
                print("Processing NumPy array from custom processing function:")
                print(pcd.shape)
                result = 'Received shape: ' + str(pcd.shape)
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

    config = {
        "host": "0.0.0.0",
        "port": 65432,
        "service_name": "3DServer",
        "service_type": "_agent._tcp.local.",
        "service_description": "PCD Processing Server",
    }

    server = ZeroconfServer(config)
    server.set_processing_function(custom_processing_function)
    server.start()
