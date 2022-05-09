## Python
1. string
   1. str.split(str="", num=string.count(str))
      1. str -- 分隔符，默认为所有的空字符，包括空格、换行(\n)、制表符(\t)
      2. num -- 分割次数。默认为 -1, 即分隔所有
   2. 

## jetbrain
1. 多行指针输入`ctrl+shift+鼠标左键`
2. 单行内容复制到下一行 `ctrl+D`

## pandas
1. 数据结构
   1. Series
   2. DataFrame

## 正则表达式
1. 量词
- `?` 匹配前面的字符0次或1次
- `*` 匹配前面的字符0次或多次
- `+` 匹配前面的字符1次或者多次
- `{m}` 匹配前面表达式m次
- `{m,}` 匹配前面表达式至少m次
- `{,n}` 匹配前面的正则表达式最多n次
- `{m,n}` 匹配前面的正则表达式至少m次，最多n次

### Python实现
1. 模块名： `re `
2. 函数
   1. re.match()
      1. 从字符串的起始位置匹配一个模式
   2. re.search()
      1. 扫描整个字符串并返回第一个成功的匹配
   3. re.findall()
   4. re.split()
      1. 根据匹配进行切割字符串，并返回列表
   5. re.compile()
   
### MySQL
1. 进入数据库 `sudo mysql -u root -p`
2. 配置文件路径 `sudo vim /etc/mysql/mysql.conf.d/mysqld.cnf`
3. 创建用户并赋予权限 
CREATE USER 'root_sql'@'localhost' IDENTIFIED BY 'yourpasswd';
GRANT ALL PRIVILEGES ON *.* TO 'root_sql'@'localhost' WITH GRANT OPTION;
FLUSH PRIVILEGES;

4. 允许远程连接，修改配置文件 `sudo vi /etc/mysql/my.cnf`
5. 重启数据库 ‘service mysql restart’


### Linux 命令
1. netstat 
   1. -t – 显示 TCP 端口。-u – 显示 UDP 端口。-n – 显示数字地址而不是主机名。-l – 仅显示侦听端口。-p – 显示进程的 PID 和名称。仅当您以 root 或 sudo 用户身份运行命令时，才会显示此信息
   2. example:  `netstat -tnlp | grep :80
