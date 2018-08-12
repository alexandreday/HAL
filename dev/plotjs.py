# 1. cp index.html and tree.js to where the json files are
# 2. start server and visualize data
import os, time

def runjs(root="./"):
    """Copies index.html and haltree.js to "root" directory and then runs
    local server to render data/javascript
    """

    run_dir = os.getcwd()
    package_dir = os.path.dirname(os.path.realpath(__file__))

    file_index = package_dir + "/js/index.html"
    file_tree = package_dir + "/js/haltree.js"
    file_server = package_dir + "/custom_server.py"
    
    out = run_dir+'/'+root
    cmd = 'cp %s %s %s %s'%(file_index, file_tree, file_server, run_dir+'/'+root)
    os.system(cmd)
    time.sleep(0.25)
    ######## ------------------ ######
    os.system('python %s/custom_server.py 1'%out)

""" if __name__ == "__main__":
    plotjs() """








