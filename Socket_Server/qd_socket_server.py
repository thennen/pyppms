import socket
import select
import sys
import string
import msvcrt
from qdcommandparser import QDCommandParser


RECEIVE_BUFFER_SIZE = 4096
PORT = 5000
ADDRESS = ''
LINE_TERM = '\r\n'

instr = QDCommandParser(sys.argv[1], line_term=LINE_TERM)

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((ADDRESS, PORT))
server_socket.listen(10)

# Dictionary to keep track of sockets and addresses.
# Keys are sockets and values are addresses.
# Add server socket to the dictionary first.
socket_dict = {server_socket: (ADDRESS, PORT)}
# socket_dict[sys.stdin] == ('keyboard', 0)

print 'Server started on port {0}.'.format(PORT)
print 'Press ESC to exit.'

keep_going = True
cmd_buffer = ''
while keep_going:
    # Get the list sockets which are ready to be read through select
    # 1 second timeout so that we can process keyboard events
    read_sockets = select.select(socket_dict.keys(), [], [], 1)[0]

    # Keyboard
    if msvcrt.kbhit():
        if ord(msvcrt.getch()) == 27:
            print 'Server exiting'
            break

    for sock in read_sockets:
        # New connection
        if sock == server_socket:
            sock_fd, address = server_socket.accept()
            socket_dict[sock_fd] = address
            print 'Client ({0}, {1}) connected.'.format(*address)
            sock_fd.send('Connected to QDInstrument socket server.' + LINE_TERM)

        # Incoming message from existing connection
        else:
            data = sock.recv(RECEIVE_BUFFER_SIZE)
            if data:
                data = data.replace('\n', '\r')
                data = data.replace('\r\r', '\r')
                cmd_buffer += data
        idx = string.find(cmd_buffer, '\r')
        if idx >= 0:
            command = cmd_buffer[:idx].upper().strip(' ')
            cmd_buffer = cmd_buffer[idx+1:]
            if command == 'EXIT':
                sock.send('Server exiting.' + LINE_TERM)
                print 'Server exiting.'
                keep_going = False
            elif command == 'CLOSE':
                sock.send('Closing connection.' + LINE_TERM)
                print 'Client ({0}, {1}) disconnected.'.format(*socket_dict[sock])
                socket_dict.pop(sock, None)
                sock.close()
            else:
                sock.send(instr.parse_cmd(command))

server_socket.close()