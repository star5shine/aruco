import socket

HOST = "0.0.0.0"
PORT = 10010

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((HOST, PORT))
s.listen(1)

print("listening on port", PORT)
conn, addr = s.accept()
print("connected from", addr)

buf = b""
while True:
    data = conn.recv(4096)
    if not data:
        print("connection closed")
        break
    buf += data
    while b"\n" in buf:
        line, buf = buf.split(b"\n", 1)
        print(line.decode("utf-8", errors="ignore"))