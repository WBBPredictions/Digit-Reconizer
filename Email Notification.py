#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 11:34:21 2018

@author: travisbarton
"""

import smtplib
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login("WBBPredictions@gmail.com", "Team6969")
 
msg = "Test"
server.sendmail("WBBPredictions@gmail.com", "sivartnotrab@gmail.com", msg)
server.quit()