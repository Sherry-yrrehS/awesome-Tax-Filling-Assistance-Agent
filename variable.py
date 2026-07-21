#全局配置
#chat模型
CHAT_MODEL_NAME="deepseek-v4-flash"

#向量模型
EMBEDDING_MODEL_NAME="text-embedding-3-small"

#Milvus连接地址
MILVUS_URI="http://localhost:19530"

#数据库名
DB_NAME="Tax_database"

#向量数据集名
COLLECTION_NAME1="Value_added_tax_policy"

#向量嵌入维度
EMBED_DIM=1536

#文件路径（Value_added_tax_policy）增值税法全文
VALUE_ADDED_TAX_POLICY_FILE1="D:\\Study\\星火杯\\7.20-7.25\\分类后文件\\顶层通用征管法规.docx"
VALUE_ADDED_TAX_POLICY_FILE2="D:\\Study\\星火杯\\7.20-7.25\\分类后文件\\分行业、特殊业务增值税专项政策.docx"
VALUE_ADDED_TAX_POLICY_FILE3="D:\\Study\\星火杯\\7.20-7.25\\分类后文件\\个体 、定期定额征收管理.docx"
VALUE_ADDED_TAX_POLICY_FILE4="D:\\Study\\星火杯\\7.20-7.25\\分类后文件\\纳税人资格与发票管理.docx"
VALUE_ADDED_TAX_POLICY_FILE5="D:\\Study\\星火杯\\7.20-7.25\\分类后文件\\申报比对与异常处理.docx"
VALUE_ADDED_TAX_POLICY_FILE6="D:\\Study\\星火杯\\7.20-7.25\\分类后文件\\增值税基础法律.docx"
VALUE_ADDED_TAX_POLICY_FILE7="D:\\Study\\星火杯\\7.20-7.25\\分类后文件\\增值税申报填报规则.docx"

VALUE_ADDED_TAX_POLICY_FILE_WORD=[
    VALUE_ADDED_TAX_POLICY_FILE1,
    VALUE_ADDED_TAX_POLICY_FILE2,
    VALUE_ADDED_TAX_POLICY_FILE3,
    VALUE_ADDED_TAX_POLICY_FILE4,
    VALUE_ADDED_TAX_POLICY_FILE5,
    VALUE_ADDED_TAX_POLICY_FILE6,
    VALUE_ADDED_TAX_POLICY_FILE7
]

VALUE_ADDED_TAX_POLICY_FILE_PDF=[
]
