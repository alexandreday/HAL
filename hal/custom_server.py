# Run python server on a different thread
import http.server
import socketserver
import sys, os, random, string, signal, subprocess
import time, random, string
import platform

# run a subprocess to create server
def random_string(n=50):
    return ''.join(random.choice(string.ascii_lowercase+string.digits) for _ in range(n))

def main():
    n_arg = len(sys.argv)
    task = sys.argv[1]

    if task == "1":
        run_cp_kill()
    elif task == "2":
        run_local_server(port = 8499)

def run_cp_kill():

    dir_path = os.path.dirname(os.path.realpath(__file__))
    html_page = dir_path+"/"+"index.html"

    # Run local server
    random_tag = random_string()
    new_tag = html_page.strip(".html") +"_"+ random_tag + ".html"
    #print(new_tag)
    os.system('cp %s %s'%(html_page, new_tag))

    # need to figure out relative copying, etc ... server should be created in same directory as .js files

    # Runs other process in the background
    p = subprocess.Popen(('python %s/custom_server.py 2'%dir_path).split(" "),cwd=dir_path) # switch directory

    time.sleep(2.0) # wait for server to launch
    url = "http://localhost:8499/%s"%("index.html".strip(".html") +"_"+ random_tag + ".html")

    if platform.system() == 'Linux':
        os.system('xdg-open %s'%url)
    elif platform.system() == 'Darwin':
        os.system('open %s'%url)
    else:
        print('OS not yet implemented.')
        raise OSError

    time.sleep(2.0)
    os.system('rm %s'%new_tag)

    # Now kill serverls
    kill_local_server(p.pid)

def run_local_server(port = 8000):
    socketserver.TCPServer.allow_reuse_address = True # required for fast reuse ! 
    """
    Check out :
    https://stackoverflow.com/questions/15260558/python-tcpserver-address-already-in-use-but-i-close-the-server-and-i-use-allow
    """
    Handler = http.server.SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer(("", port), Handler)
    print("Creating server at port", port)
    httpd.serve_forever()

def kill_local_server(pid):
    test = os.kill(pid, signal.SIGTERM) # kills subprocess, allowing clean up/clearing cache

if __name__ == "__main__":
    main()
