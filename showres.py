import os

def showres(FL, OUTNAME):
    from datamodel import GPDataSet
    import pickle
    fo=open(OUTNAME, 'w')

    FNAME=FL[0]
    ff=open(FNAME,'rb')
    ds=pickle.load(ff)
    fo.write('rowlabel, ')
    for j in ds.knob_labels:
        fo.write(j+', ')
    for j in ds.metric_labels:
        fo.write(j + ', ')
    fo.write("\n")
    ff.close()

    for FNAME in FL:
        ff=open(FNAME,'rb')
        ds=pickle.load(ff)
        #print(FNAME)
        for i in range(ds.num_previousamples):
            fo.write(str(ds.previous_rowlabels[i])+', ')
            for j in (ds.previous_knob_set[i]):
                fo.write(str(float(j)) + ', ')
            for j in (ds.previous_metric_set[i]):
                fo.write(str(float(j))+', ')
            fo.write("\n")
        ff.close()
    fo.close()

if __name__=='__main__':
    KEYdict={}
    for maindir, subdir, file_name_list in os.walk('.'):
        for filename in file_name_list:
            lf=os.path.join(maindir, filename)
            if(lf.endswith('.pkl')):
                _lf=lf.split("_")
                KEY=_lf[1]
                Round=int(_lf[2])
                CurrentRound=KEYdict.get(KEY, -1)
                if(CurrentRound==-1 or Round<CurrentRound):
                    KEYdict[KEY] = Round
                    

    totlist=[]
    for _k in KEYdict:
        fl='ds_'+_k+"_"+str(KEYdict[_k])+'_'
        showres([fl+'.pkl'], 'res_'+fl+'.txt')
        totlist.append(fl+'.pkl')

    print(totlist)
    showres(totlist, 'res_all.txt')
