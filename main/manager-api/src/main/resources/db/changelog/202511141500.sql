-- 添加声纹识别强制验证参数配置
delete from `sys_params` where id = 116;
INSERT INTO `sys_params` (id, param_code, param_value, value_type, param_type, remark)
VALUES (116, 'server.voiceprint_require_authentication', 'false', 'boolean', 1, '是否强制要求声纹验证通过才处理语音，true: 只处理已注册用户的语音，false: 所有人的语音都会被处理（默认）');

