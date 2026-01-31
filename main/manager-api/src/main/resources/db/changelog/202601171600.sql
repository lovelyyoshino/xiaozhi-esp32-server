-- 修复地点查询插件显示问题
-- 确保地点查询插件和参数正确配置

-- 1. 确保系统参数存在
DELETE FROM `sys_params` WHERE id IN (403, 404);
INSERT INTO `sys_params` (id, param_code, param_value, value_type, param_type, remark)
VALUES (403, 'plugins.get_place_info.api_key', '1ba9b66a094a5a86b22e6c7425a4f33b', 'string', 1, '高德地图API密钥');
INSERT INTO `sys_params` (id, param_code, param_value, value_type, param_type, remark)
VALUES (404, 'plugins.get_place_info.default_city', '上海', 'string', 1, '地点查询默认城市');

-- 2. 确保ai_model_provider中有地点查询插件记录
DELETE FROM `ai_model_provider` WHERE id = 'SYSTEM_PLUGIN_PLACE_INFO';
INSERT INTO ai_model_provider (id, model_type, provider_code, name, fields,
                               sort, creator, create_date, updater, update_date)
VALUES ('SYSTEM_PLUGIN_PLACE_INFO',
        'Plugin',
        'get_place_info',
        '地点查询',
        JSON_ARRAY(
                JSON_OBJECT(
                        'key', 'api_key',
                        'type', 'string',
                        'label', '高德地图 API 密钥',
                        'default', '1ba9b66a094a5a86b22e6c7425a4f33b'
                ),
                JSON_OBJECT(
                        'key', 'default_city',
                        'type', 'string',
                        'label', '默认查询城市',
                        'default', '上海'
                )
        ),
        15, 0, NOW(), 0, NOW());
