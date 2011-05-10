import gtk,os,scipy,numpy,matplotlib
CD_BASE = os.path.dirname(__file__)
class AboutBox(gtk.Window): 
    def __init__(self,version):
        super(AboutBox, self).__init__()
        about = gtk.AboutDialog()
        about.set_size_request(500, 400)
        about.set_position(gtk.WIN_POS_CENTER)
        about.set_program_name("QuTiP")
        about.set_version(version)
        about.set_license("QuTiP is licensed under the GPL3.\nSee the enclosed 'COPYING.txt' for more information.")
        about.set_copyright("Copyright (c) 2011")
        about.set_authors(['Paul D. Nation','Robert J. Johansson'])
        about.set_comments("The Quantum Toolbox in Python\nNumpy Version:      "+str(numpy.__version__)+"\nScipy Version:         "+str(scipy.__version__)+"\nMatplotlib Version:  "+str(matplotlib.__version__))
        about.set_website("http://code.google.com/p/qutip/")
        about.set_logo(gtk.gdk.pixbuf_new_from_file(str(CD_BASE)+'/logo.png'))
        about.set_icon_from_file(str(CD_BASE)+'/logo.png')
        about.run()
        about.destroy()

if __name__ == "__main__":
	AboutBox('0.1')
