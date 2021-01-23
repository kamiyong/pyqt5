"""
Created on 2021-1-12 15:44:21
@author: kamiyong
@file: Global
@description: 全局静态常量，禁止修改
"""
# 项目名称
project_name = "AI"

# 项目版本
version = "v1.2021.01.11"

# iAUTO访问IP
# ip_office
ip = '10.244.221.38'
# ip_sfc
# ip = '172.18.241.38'

# 是否上传信息到iAUTO标志位
recorder_info_to_iauto = False

# 是否下日志到本地
write_log_to_local = True

# 是否打印信息到控制台
out_msg = True

# 上传iAUTO的信息
upload_to_iauto_message = ['OK', 'no-ipad', 'no-laser', 'should-not-have-laser', 'no-camera', 'route_error']
