import pandas as pd
import numpy as np


def load_data(train_path, test_path):
    """1. 加载数据并合并，方便统一进行特征工程"""
    print("正在加载数据...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train_len = len(train)
    test['diagnosed_diabetes'] = -1

    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    print(f"数据合并完成，总行数: {len(df)}")
    return df, train_len


def handle_outliers_and_skewness(df):
    """2. 数据清洗：处理异常值与长尾分布 (V2.0 核心补丁)"""
    print("正在处理异常值和偏态分布...")

    # 补丁1：异常值盖帽 (Clipping)
    num_cols_to_clip = ['bmi', 'systolic_bp', 'diastolic_bp', 'cholesterol_total', 'ldl_cholesterol', 'triglycerides']
    for col in num_cols_to_clip:
        lower_bound = df[col].quantile(0.01)
        upper_bound = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    # 补丁2：长尾分布对数化 (Log Transform)
    skewed_cols = ['physical_activity_minutes_per_week', 'triglycerides']
    for col in skewed_cols:
        # 使用原列名加上 _log 后缀，保留原始特征供树模型使用
        df[col + '_log'] = np.log1p(df[col])

    return df


def create_features(df):
    """3. 构建所有衍生特征 (基础 + 深度交叉)"""
    print("正在构建医学与生活习惯特征...")

    # --- 基础医学交叉 ---
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    df['map_bp'] = (df['systolic_bp'] + 2 * df['diastolic_bp']) / 3
    df['tc_hdl_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1e-5)
    df['ldl_hdl_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1e-5)

    # --- 深度生活习惯交叉 (V2.0 补丁) ---
    df['lifestyle_score'] = df['diet_score'] + df['physical_activity_minutes_per_week'] / 60 - df[
        'screen_time_hours_per_day']
    df['sleep_to_screen_ratio'] = df['sleep_hours_per_day'] / (df['screen_time_hours_per_day'] + 1e-5)
    df['triglycerides_per_bmi'] = df['triglycerides'] / (df['bmi'] + 1e-5)

    # --- 连续变量分箱 ---
    age_bins = [0, 30, 45, 60, 100]
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=[0, 1, 2, 3])

    bmi_bins = [0, 18.5, 25, 30, 100]
    df['bmi_category'] = pd.cut(df['bmi'], bins=bmi_bins, labels=[0, 1, 2, 3])

    # --- 风险交互与计数 ---
    df['age_bmi_interaction'] = df['age'] * df['bmi']
    df['risk_factor_count'] = (
            (df['bmi'] > 30).astype(int) +
            (df['systolic_bp'] > 130).astype(int) +
            (df['family_history_diabetes'] == 1).astype(int) +
            (df['cholesterol_total'] > 200).astype(int)
    )

    return df


def create_group_stats(df):
    """4. 构建 Kaggle 核心：分组统计特征"""
    print("正在构建群体相对特征...")
    group_cols = ['gender', 'age_group']

    grouped_stats = df.groupby(group_cols)[['bmi', 'systolic_bp']].transform('mean')
    grouped_stats.columns = ['group_mean_bmi', 'group_mean_systolic_bp']
    df = pd.concat([df, grouped_stats], axis=1)

    df['bmi_diff_from_group'] = df['bmi'] - df['group_mean_bmi']
    df['bmi_ratio_to_group'] = df['bmi'] / (df['group_mean_bmi'] + 1e-5)

    return df


def encode_categorical(df):
    """5. 类别特征编码 (V2.0 严格化)"""
    print("正在进行类别编码...")

    # 二元变量
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

    # 有序类别 (Ordinal Encoding)
    smoking_mapping = {'Never': 0, 'Former': 1, 'Current': 2}
    income_mapping = {'Low': 1, 'Lower-Middle': 2, 'Middle': 3, 'Upper-Middle': 4, 'High': 5}
    edu_mapping = {'Highschool': 1, 'Bachelors': 2, 'Graduate': 3, 'Postgraduate': 4}

    df['smoking_status'] = df['smoking_status'].map(smoking_mapping)
    df['income_level_encoded'] = df['income_level'].map(income_mapping)
    df['education_level_encoded'] = df['education_level'].map(edu_mapping)

    # 将分箱生成的 category 类型转为 float，方便后续输入模型
    for col in ['age_group', 'bmi_category']:
        df[col] = df[col].astype(float)

    return df


def split_and_save(df, train_len):
    """6. 分离并保存最终数据"""
    print("正在分离数据集并保存...")
    train_engineered = df.iloc[:train_len].copy()
    test_engineered = df.iloc[train_len:].copy()
    test_engineered.drop(['diagnosed_diabetes'], axis=1, inplace=True)

    # 存为 V2 版本的 CSV，方便我们后续做效果对比
    train_engineered.to_csv('train_engineered_v2.csv', index=False)
    test_engineered.to_csv('test_engineered_v2.csv', index=False)
    print(f"🎉 特征工程 V2.0 处理完成！")
    print(f"训练集形状: {train_engineered.shape} | 测试集形状: {test_engineered.shape}")


if __name__ == "__main__":
    train_file = 'train.csv'
    test_file = 'test.csv'

    # 执行流水线
    df_all, length = load_data(train_file, test_file)
    df_all = handle_outliers_and_skewness(df_all)
    df_all = create_features(df_all)
    df_all = create_group_stats(df_all)
    df_all = encode_categorical(df_all)
    split_and_save(df_all, length)