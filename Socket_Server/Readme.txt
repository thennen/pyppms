Readme for Quantum Design Socket Server

#### Introduction ####
This server provides access to the set and get methods for temperature, field, and chamber on four QD platforms:
PPMS, DynaCool, VersaLab, and MPMS3. The server is implemented in Python, and runs on the same computer as MultiVu.
Clients connect to the server using sockets. Socket connections can be made from telnet or programs you write and
can be implemented in many operating systems.

This is a very simple server, intended for expert use. The Python source is included, so feel free to modify it to
suit your needs.

The server interacts with MultiVu through its OLE interface--see the file qdinstrument.py. With knowledge of MultiVu's
OLE methods you can add more capability. Also, you can uses qdinstrument.py to send commands to MultiVu from
Python directly.

#### Getting Started ####
To get started, install a Python 2.7 distribution. Quantum Design recommends Anaconda (http://continuum.io/downloads)
because it includes everything needed for this server and other packages useful for scientific computing. However, any
Python 2.7 distribution that includes msvcrt, pythoncom, and win32com will work.

Before starting the server, be sure that MultiVu is running. For testing purposes, you can use MultiVu in simulation
mode on an office computer.

To start the server, double-click the batch file for your platform type. This opens a console window indicating
that the server has started. To stop the server, press the escape key (make sure the server console window has focus).

To connect to the server from a client machine, telnet to the ip address of the server computer on port 5000. You will
get the response "Connected to QDInstrument socket server." You can then type commands and verify that communication
with MultiVu is working. A simple example is:
TEMP?
This returns something like "0, 300.0, 1" where 0 is a return code, 300.0 is the temperature, and 1 is the temperature
status (stable).

#### Command List ####
In the following command list, commands are shown in all caps, but the server is case-independent. Argument and
return numerical values are shown in lower case. One or more spaces must be inserted between the command and arguments.
Arguments are comma-separated with or without spaces. The return codes can generally be ignored, but may be useful for
troubleshooting. Below each command is the response. For a listing of status and state codes, see the
section "Quantum Design MultiVu OLE Methods" in application node 1070-209.

TEMP?
return_code, temperature, status

TEMP temperature_setpoint, rate, mode
return_code

FIELD?
return_code, field, status

FIELD field_setpoint, rate, approach_mode, end_mode
return_code

CHAMBER?
return_code, state

CHAMBER code
return_code

The following commands do not interact with MultiVu, but are for the server:
CLOSE   This closes the socket connection, leaving the server running (other programs can still connect).
EXIT    This causes the server to exit. No further connections are possible without restarting the server.

#### Troubleshooting ####
Typical connection issues are due to:
- Firewall. You might need to allow connections on port 5000 in your firewall.
- Port conflict. If port 5000 is in use, open the file qd_socket_server.py and change the PORT value one with
no conflict.
