#!/usr/bin/env python

# This is a very crude script for plotting graphs based on rtm-bench logs.
#
# Copyright (c) 2013 Carl G. Ritson <cgr@kent.ac.uk>
# 
# This file may be freely used, copied, or distributed without compensation 
# or licensing restrictions, but is done so without any warranty or 
# implication of merchantability or fitness for any particular purpose.
# 

import re, sys
import copy
import cPickle as pickle

can_plot = False
try:
    import matplotlib
    matplotlib.use('Agg')

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    
    can_plot = True
except:
    can_plot = False

def warn(*s):
    print >>sys.stderr, "".join(map(str, s))

def die(*s):
    warn(*s)
    sys.exit(1)

def line_styles():
    #colours = ['#7fc97f','#beaed4','#fdc086','#a6cee3','#1f78b4','#b2df8a','#66c2a5','#fc8d62','#8da0cb','#1f78b4']
    colours = ['#404040', '#808080', '#000000', '#606060', '#808080']
    styles = ['-', '-', ':', '--', ':']
    markers = ['', '', '', '', '']
    widths = [1.5, 1.5, 1.4, 1.4, 1.0]
    result = []
    for i in range(5):
        result.append({
            'c': colours[i],
            'ls': styles[i],
            'm': markers[i],
            'lw': widths[i]
        })
    result.reverse()
    return result * 4

def counter_names():
    return {
        0: 'unknown',
        1: 'xabort',
        2: 'conflict',
        4: 'conflict',
        6: 'conflict-retry',
        8: 'overflow',
        14: 'conflict-ovfw-retry',
        41: 'failure',
        42: 'success'
    }
def counter_order():
    return [42, 41, 0, 2, 4, 6, 14, 8]
def ordered_counters(counters):
    order = counter_order()
    result = []
    for cv in order:
        if cv in counters:
            result.append((cv, counters[cv]))
    return result

def new_test_entry(mode=None, thread=None, n_threads=1):
    entry = {
        'test': None,
        'thread': int(thread),
        'n_threads': n_threads,
        #'mode': mode,
        'count': 0,
        'op_size': 0,
        'stride': 0,
        'ns': 0,
        'cycles': 0,
        'counters': {}
    }
    return entry

def dup_entry(entry):
    return copy.deepcopy(entry)

def enhance_entry(entry):
    if entry['ns'] > 0:
        entry['bpns'] = float(entry['op_size'] * entry['count']) / float(entry['ns'])
    else:
        entry['bpns'] = 0.0
    if entry['op_size'] > 0:
        entry['nspb'] = float(entry['ns']) / float(entry['op_size'] * entry['count']) 
    else:
        entry['nspb'] = 0.0
    
    pcc = {}
    for (c, v) in entry['counters'].items():
        pcc[c] = (float(v) / float(entry['count'])) * 100.0
    entry['pcc'] = pcc

def parse_test(mode, threads):
    test_line = re.compile(r'test = (\S+), count = (\d+), op_size = (\d+), stride = (\d+)')
    mode_line = re.compile(r'[ux]_(read|write|cas|abortn|abortm)')
    timing_line = re.compile(r'ns = (\d+), cycles = (\d+)')
    counter_line = re.compile(r'counter (\d+) = (\d+)')
    
    data = {}
    thread_mode = {}
    n_threads = len(threads.keys())
    for (thread, lines) in threads.items():
        data[thread] = [] 
        entry = new_test_entry(mode=mode, thread=thread, n_threads=n_threads)
        for line in lines:
            if test_line.match(line):
                if entry['test'] is not None:
                    enhance_entry(entry)
                    data[thread].append(entry)

                m = test_line.match(line)
                entry = new_test_entry(mode=mode, thread=thread, n_threads=n_threads)
                entry['test'] = m.group(1)
                entry['count'] = int(m.group(2))
                entry['op_size'] = int(m.group(3))
                entry['stride'] = int(m.group(4))
            elif timing_line.match(line):
                m = timing_line.match(line)
                entry['ns'] = int(m.group(1))
                entry['cycles'] = int(m.group(2))
            elif counter_line.match(line):
                m = counter_line.match(line)
                entry['counters'][int(m.group(1))] = int(m.group(2))
            elif mode_line.match(line):
                thread_mode[thread] = line
        
        if entry['test'] is not None:
            enhance_entry(entry)
            data[thread].append(entry)

        if len(data[thread]) == 0:
            del data[thread]

    key = []
    for thread in range(n_threads):
        if (thread in data) and (len(data[thread]) > 0):
            key.append(thread_mode[thread])

    return (key, data)

def parse_log(fn):
    fh = open(fn, 'r')
    lines = fh.readlines()
    fh.close()
    
    mode = None
    memory = "isolated"
    mode_line = re.compile(r'(single|homogenous|heterogenous) thread tests')
    memory_line = re.compile(r'(isolated|shared) memory tests')
    thread_line = re.compile(r'(\d+): (.+)')
    thread_mode_line = re.compile(r'(\d+): ([ux]_)?(read|write|cas|sleeper|abortn|abortm)')
    
    buffer = {}
    data = []
    in_reset = False

    for line in lines:
        if mode_line.match(line):
            if len(buffer.keys()) > 0:
                (key, results) = parse_test(mode, buffer)
                data.append(([mode,memory] + key, results))
            buffer = {}
            in_reset = True
            
            m = mode_line.match(line)
            mode = m.group(1)
        elif thread_mode_line.match(line):
            m = thread_mode_line.match(line)
            thread = int(m.group(1))
            if not in_reset:
                if len(buffer.keys()) > 0:
                    (key, results) = parse_test(mode, buffer)
                    data.append(([mode,memory] + key, results))
                buffer = {}
                in_reset = True
        elif memory_line.match(line):
            m = memory_line.match(line)
            memory = m.group(1)

        if thread_line.match(line):
            m = thread_line.match(line)
            thread = int(m.group(1))
            i_line = m.group(2)
            if not (thread in buffer):
                buffer[thread] = []
            buffer[thread].append(i_line)
            if in_reset and len(buffer[thread]) >= 3:
                in_reset = False
    
    return data


def plot_entries(pages, data, threads, rate_title, rate_keys, error_title, error_keys):
    mt = len(threads) > 1

    # combined rate graph
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ls = line_styles()
    for thread in threads:
        for key in rate_keys:
            if mt:
                label = "%d:%s" % (thread, key)
            else:
                label = key
            xs = []
            ys = []
            for entry in data:
                if (entry['thread'] == thread) and (entry['test'] == key):
                    xs.append(entry['op_size'])
                    ys.append(entry['bpns'])

            if len(xs) > 0 and len(ys) > 0:
                style = ls.pop()
                ax.plot(xs, ys, c=style['c'], ls=style['ls'], marker=style['m'], label=label, lw=style['lw'])
            else:
                print rate_keys, thread, label, 'no data'

    if rate_title is not None:
        ax.set_title(rate_title)
    #ax.set_ylim(min_y, max_y)
    ax.set_ylabel('bytes/ns')
    ax.set_xlabel('bytes/op')
    ax.legend(loc='best', ncol=2)
    
    fig.savefig(pages, format='pdf')
    
    # error graphs
    for thread in threads:
        for key in error_keys:
            if mt:
                label = "%d:%s" % (thread, key)
            else:
                label = key
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            
            counters = {}
            names = counter_names()
            for entry in data:
                if (entry['thread'] == thread) and (entry['test'] == key):
                    for c in entry['pcc'].keys():
                        if c not in counters:
                            counters[c] = names[c]
            
            ls = line_styles()
            for (cv, cn) in ordered_counters(counters):
                xs = []
                ys = []
                for entry in data:
                    if (entry['thread'] == thread) and (entry['test'] == key):
                        xs.append(entry['op_size'])
                        if cv in entry['pcc']:
                            ys.append(entry['pcc'][cv])
                        else:
                            ys.append(0.0)
                if len(xs) > 0 and len(ys) > 0:
                    style = ls.pop()
                    ax.plot(xs, ys, c=style['c'], ls=style['ls'], marker=style['m'], label=cn, lw=style['lw'])
                else:
                    print error_keys, thread, key, cv, cn, 'no data'

            if error_title is not None:
                ax.set_title(error_title + ' ' + label)
            ax.set_ylim(0.0, 100.0)
            ax.set_ylabel('%')
            ax.set_xlabel('bytes/op')
            ax.legend(loc='best', ncol=1)
            
            fig.savefig(pages, format='pdf')

def select_data(key, data, precise = False):
    selected = []
    for (k, v) in data:
        if len(key) <= len(k):
            match = (not precise) or (len(key) == len(k))
            for i in range(len(key)):
                if k[i] != key[i]:
                    match = False
            if match:
                selected.append((k, v))
    return selected

def add_entry(dst, src):
    if 'cn' in dst:
        dst['cn'] += 1
    else:
        dst['cn'] = 2

    dst['count'] += src['count']
    dst['cycles'] += src['cycles']
    dst['ns'] += src['ns']
    for (k, v) in src['counters'].items():
        if k in dst['counters']:
            dst['counters'][k] += v
        else:
            dst['counters'][k] = v

def avg_entry(entry):
    enhance_entry(entry)

def sum_data(data):
    master = {}
    for (k, v) in data:
        kn = "|".join(k)
        for (tn, td) in v.items():
            if kn in master:
                md = master[kn]
                for i in range(len(td)):
                    add_entry(md[i], td[i])
            else:
                master[kn] = dup_entry(td)
    result = []
    for (k, v) in master.items():
        for entry in v:
            avg_entry(entry)
            result.append(entry)
    return result

def flatten_data(data):
    result = []
    for (k, v) in data:
        for (tn, td) in v.items():
            result.extend(td)
    return result

def cl_data(data):
    result = []
    for entry in data:
        if (entry['op_size'] % 64) == 0:
            result.append(entry)
    return result

def lt_data(n, data):
    result = []
    for entry in data:
        if (entry['op_size'] <= n):
            result.append(entry)
    return result

def report_entry(entry):
    print ", ".join(map(str,[entry['test'], entry['op_size'], float(entry['ns']) / float(entry['count']), float(entry['cycles']) / float(entry['count'])]))

def plot_data(fn, data):
    ops_read = ['x_read32', 'x_read64', 'u_read32', 'u_read64']
    ops_write = ['x_write32', 'x_write64', 'u_write32', 'u_write64']
    ops_cas = ['x_cas32', 'x_cas64', 'u_cas32', 'u_cas64']
    ops_wc = ['u_write64', 'x_write64', 'x_cas64']
    ops_rw32 = ['x_read32', 'x_write32', 'x_cas32']
    ops_rw64 = ['x_read64', 'x_write64', 'x_cas64']
    ops_abort32 = ['x_abortn32', 'x_abortm32', 'x_cas32']
    ops_abort64 = ['x_abortn64', 'x_abortm64', 'x_cas64']

    st = sum_data(select_data(['single', 'isolated'], data))
    st_64 = cl_data(st)
    st_lt300 = lt_data(280, st)
    st_lt1000 = lt_data(1000, st)

    for entry in st:
        if entry['op_size'] <= 64:
            report_entry(entry)

    mt_xrxr = lt_data(1000, flatten_data(select_data(['homogenous', 'shared', 'x_read', 'x_read'], data, precise=True)))
    mt_xwxw = lt_data(1000, flatten_data(select_data(['homogenous', 'shared', 'x_write', 'x_write'], data, precise=True)))
    mt_ucuc = lt_data(1000, flatten_data(select_data(['homogenous', 'shared', 'u_cas', 'u_cas'], data, precise=True)))
    mt_xcxc = lt_data(1000, flatten_data(select_data(['homogenous', 'shared', 'x_cas', 'x_cas'], data, precise=True)))
    mt_xcur = lt_data(1000, flatten_data(select_data(['heterogenous', 'shared', 'x_cas', 'u_read'], data, precise=True)))
    mt_xwuw = lt_data(1000, flatten_data(select_data(['heterogenous', 'shared', 'x_write', 'u_write'], data, precise=True)))

    pages = PdfPages(fn)
    plot_entries(pages, st_64, [0], None, ops_read, '', ops_read)
    plot_entries(pages, st_64, [0], None, ops_write, '', ops_write)
    plot_entries(pages, st_64, [0], None, ops_cas, '', ops_cas)
    plot_entries(pages, st_64, [0], None, ops_abort32, None, [])
    plot_entries(pages, st_64, [0], None, ops_abort64, None, [])
    plot_entries(pages, st_lt1000, [0], None, ops_read, None, [])
    plot_entries(pages, st_lt1000, [0], None, ops_write, None, [])
    plot_entries(pages, st_lt1000, [0], None, ops_cas, None, [])
    plot_entries(pages, st_lt1000, [0], None, ops_rw32, None, [])
    plot_entries(pages, st_lt1000, [0], None, ops_rw64, None, [])
    plot_entries(pages, st_lt1000, [0], None, ops_abort64, None, [])
    plot_entries(pages, st_lt300, [0], None, ops_read, None, [])
    plot_entries(pages, st_lt300, [0], None, ops_write, None, [])
    plot_entries(pages, st_lt300, [0], None, ops_cas, None, [])
    plot_entries(pages, st_lt300, [0], None, ops_rw32, None, [])
    plot_entries(pages, st_lt300, [0], None, ops_rw64, None, [])
    plot_entries(pages, st_lt300, [0], None, ops_abort64, None, [])
    plot_entries(pages, mt_xrxr, [0,1], None, ['x_read32'], '', ['x_read32'])
    plot_entries(pages, mt_xwxw, [0,1], None, ['x_write32'], '', ['x_write32'])
    plot_entries(pages, mt_xcxc, [0,1], None, ['x_cas32'], '', ['x_cas32'])
    plot_entries(pages, mt_xcur, [0,1], None, ['x_cas32', 'u_read32'], '', ['x_cas32', 'u_read32'])
    plot_entries(pages, mt_xwuw, [0,1], None, ['x_write32', 'u_write32'], '', ['x_write32', 'u_write32'])
    plot_entries(pages, mt_ucuc + mt_xcxc, [0,1], None, ['x_cas32', 'u_cas32'], None, [])
    pages.close()

def write_data(fn, data):
    fh = open(fn, 'wb')
    pickle.dump(data, fh)
    fh.close()

def read_data(fn):
    fh = open(fn, 'rb')
    data = pickle.load(fh)
    fh.close()
    return data

def main(args):
    if len(args) < 2:
        die('rtm-graph.py <log-file> <output-file>') 

    log_file = args[0]
    out_file = args[1]

    if not can_plot:
        warn('unable to plot')

    print 'parsing', log_file
    data = parse_log(log_file)
    
    for (name, entries) in data:
        print name

    print 'plotting graphs to', out_file
    if can_plot:
        plot_data(out_file, data)

if __name__ == "__main__":
    main(sys.argv[1:])
