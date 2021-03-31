# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 19:50:23 2021

@author: mathe
"""

import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)  