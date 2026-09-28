import re,sys,struct,hashlib
t=open(sys.argv[1]).read()
for name in ["cwsr_trap_gfx12_hex","cwsr_trap_gfx12_1_0_hex"]:
    m=re.search(r"%s\[\]\s*=\s*\{(.*?)\};"%name,t,re.S)
    words=[int(x,16) for x in re.findall(r"0x[0-9a-fA-F]+",m.group(1))]
    b=struct.pack("<%dI"%len(words),*words)
    print(name,len(b),hashlib.sha256(b).hexdigest())
    if len(sys.argv)>2 and name.endswith("1_0_hex"): open(sys.argv[2],"wb").write(b)
