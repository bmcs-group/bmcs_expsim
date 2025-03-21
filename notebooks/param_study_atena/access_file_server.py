from smbprotocol.connection import Connection
from smbprotocol.session import Session
from smbprotocol.tree import TreeConnect
from smbprotocol.open import Open, CreateDisposition

def access_file_server(username, password):
    # Define the Samba server and share details
    server_name = "fileserver.imb.rwth-aachen.de"
    share_name = "data"

    # Establish a connection to the server
    connection = Connection(username="client", server=server_name, port=445)
    connection.connect()

    # Create a session with the server using the provided credentials
    session = Session(connection, username=username, password=password)
    session.connect()

    # Connect to the specific share on the server
    tree = TreeConnect(session, f"\\\\{server_name}\\{share_name}")
    tree.connect()

    # Open the root directory of the share
    root_dir = Open(tree, "", access_mask=0x00120089, create_disposition=CreateDisposition.FILE_OPEN)
    root_dir.create()

    # List the contents of the root directory
    for file_info in root_dir.query_directory("*"):
        print(f"Name: {file_info['file_name']}, Is Directory: {file_info['is_directory']}")

    # Clean up and close the connections
    root_dir.close()
    tree.disconnect()
    session.disconnect()
    connection.disconnect()

