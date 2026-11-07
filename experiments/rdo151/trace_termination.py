"""Instrument a temporary native encoder; reference decoder/artifact remain unchanged."""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import profile_reference as pr

OUTPUT=None


def build_native(directory):
    source=pr.RUNTIME/'runtime/entropy/rc64_encode_backend.c'
    text=source.read_text()
    includes='''
#include <stdio.h>
static FILE *trace_events=NULL,*trace_groups=NULL;
static int trace_frame=-1,trace_group=-1;
void trace_open(const char *events,const char *groups) {
 trace_events=fopen(events,"w"); trace_groups=fopen(groups,"w");
 if(trace_events) fprintf(trace_events,"frame,group,symbol_index,target,frequency,below,low_before,high_before,low_after,high_after,emitted_before,emitted_after,pending_before,pending_after,e1,e2,e3,p0,p1,p2,p3,p4\\n");
 if(trace_groups) fprintf(trace_groups,"frame,group,count,low,high,emitted,pending\\n");
}
void trace_set(int frame,int group) {trace_frame=frame;trace_group=group;}
void trace_close(void) {if(trace_events){fclose(trace_events);trace_events=NULL;} if(trace_groups){fclose(trace_groups);trace_groups=NULL;}}
'''
    old='        const float *row = probabilities + index * RC64_ALPHABET;'
    new='''        uint64_t trace_low=e->low,trace_high=e->high;
        size_t trace_bits=e->bits,trace_pending=e->pending;
        unsigned trace_e1=0,trace_e2=0,trace_e3=0;
'''+old
    replacements={old:new,
        'if (enc_emit(e, 0u)) return -7;':'trace_e1++; if (enc_emit(e, 0u)) return -7;',
        'if (enc_emit(e, 1u)) return -7;':'trace_e2++; if (enc_emit(e, 1u)) return -7;',
        '                e->pending++;':'                trace_e3++; e->pending++;',
        '''    }
    return 0;
}

int rc64_encoder_finish''':'''        if(trace_events && trace_frame>=430 && trace_frame<=432 && (trace_e1||trace_e2||trace_e3)) {
         fprintf(trace_events,"%d,%d,%zu,%d,%u,%llu,%llu,%llu,%llu,%llu,%zu,%zu,%zu,%zu,%u,%u,%u,%.9g,%.9g,%.9g,%.9g,%.9g\\n",trace_frame,trace_group,index,target,frequencies[target],(unsigned long long)below,(unsigned long long)trace_low,(unsigned long long)trace_high,(unsigned long long)e->low,(unsigned long long)e->high,trace_bits,e->bits,trace_pending,e->pending,trace_e1,trace_e2,trace_e3,row[0],row[1],row[2],row[3],row[4]);
        }
    }
    if(trace_groups && trace_frame>=430) {
     fprintf(trace_groups,"%d,%d,%zu,%llu,%llu,%zu,%zu\\n",trace_frame,trace_group,count,(unsigned long long)e->low,(unsigned long long)e->high,e->bits,e->pending);
     fflush(trace_groups);
    }
    return 0;
}

int rc64_encoder_finish'''}
    for old,new in replacements.items():
        if text.count(old)!=1:raise ValueError('native source differs; refusing instrumentation')
        text=text.replace(old,new)
    # Absolute include ensures inherited decoder definitions resolve without altering source.
    text=text.replace('#include "rc64_backend.c"','#include '+json.dumps(str(source.parent/'rc64_backend.c')))
    wrapper=directory/'trace.c'
    wrapper.write_text(includes+text+'''
uint64_t profile_pending(void *v) { return ((rc64_encoder *)v)->pending; }
uint64_t profile_emitted(void *v) { return ((rc64_encoder *)v)->bits; }
''')
    lib=directory/'rc64.so';corrector=directory/'corrector.so'
    cc=os.environ.get('CC','cc')
    common=[cc,'-O3','-std=c11','-shared','-fPIC','-ffp-contract=off','-fno-fast-math']
    subprocess.run(common+[str(wrapper),'-o',str(lib),'-lm'],check=True)
    subprocess.run(common+[str(pr.RUNTIME/'runtime/f26_corrector_native.c'),'-o',str(corrector),'-lm'],check=True)
    return lib,corrector


class TraceProfile(pr.Profile):
    def frame_start(self,frame):
        super().frame_start(frame);self.trace_group_index=0

    def group(self,rows,symbols):
        super().group(rows,symbols)
        lib=self.shadow.library
        if not hasattr(self,'trace_lib'):
            self.trace_lib=lib
            lib.trace_open.argtypes=[ctypes.c_char_p,ctypes.c_char_p];lib.trace_open.restype=None
            lib.trace_set.argtypes=[ctypes.c_int,ctypes.c_int];lib.trace_set.restype=None
            lib.trace_close.argtypes=[];lib.trace_close.restype=None
            lib.trace_open(os.fsencode(self.args.output/'termination_events.csv'),os.fsencode(self.args.output/'termination_groups.csv'))
        lib.trace_set(self.frame,self.trace_group_index);self.trace_group_index+=1

    def finish(self,info):
        self.trace_lib.trace_close()
        result=super().finish(info)
        import csv
        path=self.args.output/'termination_events.csv'
        last=None;counts={'e1':0,'e2':0,'e3':0}
        with path.open() as f:
            for row in csv.DictReader(f):
                for key in counts:counts[key]+=int(row[key])
                if int(row['emitted_after'])>int(row['emitted_before']):last=row
        summary=dict(last_emission_in_traced_window=last,event_counts_frames_430_to_432=counts,
            symbol_index_scope='index within native encode batch; group is causal decoder batch order',
            full_stream_reproduced=result['shadow_shortened_matches_published'],
            token_parity=result['all_profiled_tokens_match_cache'])
        (self.args.output/'termination_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary,indent=2),flush=True)
        return result

if __name__=='__main__':
    pr.build_native=build_native
    pr.Profile=TraceProfile
    pr.main()
