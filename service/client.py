import socket
import json

def send_request(args):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('localhost', 9999))

    request = json.dumps(args)
    client.send(request.encode('utf-8'))

    response = client.recv(4096).decode('utf-8')
    print(f"Server response: {response}")
    client.close()

if __name__ == "__main__":
    args = {
        "tensor_size": 100,
        "communication": False,
        "device": "cpu",
        "world_size": 2,
    }

    send_request(args)
