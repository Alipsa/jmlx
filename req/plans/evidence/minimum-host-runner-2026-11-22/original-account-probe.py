import os,json,pwd,grp,hashlib,pathlib,subprocess,shutil,platform
home=pathlib.Path(pwd.getpwuid(os.getuid()).pw_dir); runner=home/'actions-runner'
clean={'PATH':'/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin','HOME':str(home),'USER':'jmlxrunner','LOGNAME':'jmlxrunner'}
def command(*args):return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT,env=clean).strip()
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
result={'uid':os.getuid(),'user':pwd.getpwuid(os.getuid()).pw_name,'homeMode':oct(home.stat().st_mode&0o777),'groups':command('/usr/bin/id','-Gn').split(),'host':{'swVers':command('/usr/bin/sw_vers'),'architecture':platform.machine(),'virtualized':command('/usr/sbin/sysctl','-n','kern.hv_vmm_present')},'runner':{},'savedEnvironment':{},'isolation':{},'tools':{}}
config=json.loads((runner/'.runner').read_text(encoding='utf-8-sig'));result['runner']={k:config.get(k) for k in ('agentId','agentName','workFolder')};result['runner']['listenerVersion']=command(str(runner/'bin/Runner.Listener'),'--version');result['runner']['listenerSha256']=digest(runner/'bin/Runner.Listener')
for name in ('.env','.path'):
 b=(runner/name).read_bytes();result['savedEnvironment'][name]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'containsOwnerHome':b'/Users/per' in b,'mode':oct((runner/name).stat().st_mode&0o777)}
for name,path in [('ownerGradleProperties','/Users/per/.gradle/gradle.properties'),('ownerSshDirectory','/Users/per/.ssh'),('ownerSigningDirectory','/Users/per/.gnupg')]:result['isolation'][name]={'readable':os.access(path,os.R_OK),'writable':os.access(path,os.W_OK),'traversable':os.access(path,os.X_OK)}
for name,args in [('python3',['--version']),('git',['--version'])]:
 executable=shutil.which(name,path=clean['PATH']);p=pathlib.Path(executable).resolve();st=p.stat();result['tools'][name]={'selected':executable,'resolved':str(p),'uid':st.st_uid,'gid':st.st_gid,'mode':oct(st.st_mode&0o777),'version':command(executable,*args)}
print(json.dumps(result,indent=2))
