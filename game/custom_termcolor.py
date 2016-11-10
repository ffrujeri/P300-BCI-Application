from __future__ import print_function

from termcolor import colored
import os

################################ termocolor functions ################################################
	#pour python2 il faut importer la function print de python3 from __future__ import print_function


if os.name == 'nt': # dans os windows il faut le module colorama en plus
    import colorama
    colorama.init()


#_______________________________________________________________________________
def custom_print_in_blue(*args, **kwargs):
    args = list(args)
    output = list()
    if os.name == 'nt':
        color_blue = 'cyan'
    else:
        color_blue = 'blue'
    
    for index, arg in enumerate(args):
        if not isinstance(arg, basestring):
            args[index] = str(arg)
            output.append(colored(args[index], color_blue, attrs=['bold']))
        else:
            output.append(colored(args[index], color_blue))
    print(' '.join(output), **kwargs)

#_______________________________________________________________________________
def custom_print_in_green(*args):
    args = list(args)
    output = list()
    for index, arg in enumerate(args):
        if not isinstance(arg, basestring):
            args[index] = str(arg)
            output.append(colored(args[index], 'green', attrs=['bold']))
        else:
            output.append(colored(args[index], 'green'))
    print(' '.join(output))
#_______________________________________________________________________________
def custom_print_in_red(*args):
    args = list(args)
    output = list()
    for index, arg in enumerate(args):
        if not isinstance(arg, basestring):
            args[index] = str(arg)
            output.append(colored(args[index], 'red', attrs=['bold']))
        else:
            output.append(colored(args[index], 'red'))
    print(' '.join(output))
#_______________________________________________________________________________
def custom_print_in_yellow(*args):
    args = list(args)
    output = list()
    for index, arg in enumerate(args):
        if not isinstance(arg, basestring):
            args[index] = str(arg)
            output.append(colored(args[index], 'yellow', attrs=['bold']))
        else:
            output.append(colored(args[index], 'yellow'))
    print(' '.join(output))


################################ termocolor functions #####################################################################
