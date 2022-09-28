import dill

auprc = dill.load(open("/hpf/largeprojects/davidm/blaverty/classify_lfs/output/all/rf_auprc", "rb"))

print(auprc)
