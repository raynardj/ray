# aria2

## How to online bit-torrent

### Server Side

Download aria2 on server

```bash
sudo apt install aria2
mkdir aria2
mkdir ardl
```

Create a file on path:

```
~/aria2/aria2.session
```

Create a config file on ```/etc/aria2.conf```

Change the token in it
```bash
#允许rpc

enable-rpc=true

#允许所有来源, web界面跨域权限需要

rpc-allow-origin-all=true

#允许非外部访问

rpc-listen-all=true

#RPC端口, 仅当默认端口被占用时修改

#rpc-listen-port=6800


rpc-secret={$your_token_here}
#用户名

# rpc-user=ray

#密码

# rpc-passwd=sbswdsbszdywaqll



###速度相关

#最大同时下载数(任务数), 路由建议值: 3

max-concurrent-downloads=5

#断点续传

continue=true

#同服务器连接数

max-connection-per-server=5

#最小文件分片大小, 下载线程数上限取决于能分出多少片, 对于小文件重要

min-split-size=20M

#单文件最大线程数, 路由建议值: 5

split=10

#下载速度限制 0 不限制

# 1mb downlaod limit
max-overall-download-limit=1000000

#单文件速度限制

max-download-limit=0

#上传速度限制

# 100k upload limit
max-overall-upload-limit=100000

#单文件速度限制

max-upload-limit=0

#断开速度过慢的连接

#lowest-speed-limit=0

#验证用，需要1.16.1之后的release版本

#referer=*



###进度保存相关

input-file=/home/salvor/aria2/aria2.session

save-session=/home/salvor/aria2/aria2.session

#定时保存会话，需要1.16.1之后的release版

#save-session-interval=60



###磁盘相关

#文件保存路径, 默认为当前启动位置

dir=/home/salvor/aria2

#文件缓存, 使用内置的文件缓存, 如果你不相信Linux内核文件缓存和磁盘内置缓存时使用, 需要1.16及以上版本

#disk-cache=0

#另一种Linux文件缓存方式, 使用前确保您使用的内核支持此选项, 需要1.15及以上版本

#enable-mmap=true

#文件预分配, 能有效降低文件碎片, 提高磁盘性能. 缺点是预分配时间较长

#所需时间 none < falloc ? trunc << prealloc, falloc和trunc需要文件系统和内核支持

file-allocation=prealloc



###BT相关

#启用本地节点查找

bt-enable-lpd=true

#添加额外的tracker

#bt-tracker=<URI>,…

#单种子最大连接数

#bt-max-peers=55

#强制加密, 防迅雷必备

#bt-require-crypto=true

#当下载的文件是一个种子(以.torrent结尾)时, 自动下载BT

follow-torrent=true

#BT监听端口, 当端口屏蔽时使用

#listen-port=6881-6999

#aria2亦可以用于PT下载, 下载的关键在于伪装

#不确定是否需要，为保险起见，need more test

enable-dht=true

bt-enable-lpd=true

enable-peer-exchange=true

#修改特征

user-agent=uTorrent/2210(25130)

peer-id-prefix=-UT2210-

#修改做种设置, 允许做种

seed-ratio=0

#保存会话

force-save=true

bt-hash-check-seed=true

bt-seed-unverified=true

bt-save-metadata=true

#定时保存会话，需要1.16.1之后的某个release版本

#save-session-interval=60

```

Notice the follow has to be enabled:

```bash
enable-dht=true

bt-enable-lpd=true

enable-peer-exchange=true
```
Add the following line to ```~/.bashrc```, wrap up the command "ar"

```bash
alias ar='aria2c --conf-path=/etc/aria2.conf'
```

run ```source ~/.bashrc```

```bash
tmux new -s ar
ar
```

### Local

Setup an apache2 server, containing ```WEBUI``` repo, [webui git address](https://github.com/ziahamza/webui-aria2)

```
git clone https://github.com/ziahamza/webui-aria2.git
```
rename the folder to ```webui```

open localhost/xxx/webui in browser

set the config in "settings" > "connection setting"

### Download online file to local

Put downloaded files to an apache server, we can have and download address

at local we do:
```bash
aria2c -x4 http://download_address
```
for downloading the address http://download_address in 4 connection/host