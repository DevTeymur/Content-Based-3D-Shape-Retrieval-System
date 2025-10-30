import cec
import socket
import json

cec.init()
tv = cec.Device(0)

HOST = '127.0.0.1'  # localhost
PORT = 65432        # any free port

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"CEC service running on {HOST}:{PORT}")

    while True:
        conn, addr = s.accept()
        with conn:
            data = conn.recv(1024).decode().strip()
            if data == "status":
                status = tv.power_status
                response = "on" if status == cec.CEC_POWER_STATUS_ON else "off"
                conn.sendall(response.encode())
            else:
                conn.sendall(b"unknown command")