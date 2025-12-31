

def format_vector(vector, decimal_places=2, format="decimal"):
    # --------
    if format == "decimal":
        # 通常の小数形式で指定された小数点以下の桁数で四捨五入
        formatted_vector = [f"{round(v, decimal_places):.{decimal_places}f}" for v in vector]
    elif format == "decimal_exp":
        # 指定された小数点以下の桁数で指数形式に変換
        formatted_vector = [f"{v:.{decimal_places}e}" for v in vector]
    else:
        raise NotImplementedError()
    # --------
    return  "[" + ", ".join(formatted_vector) + "]"


if __name__ == '__main__':
    # 例の入力
    rotation_vector = [1.12345, -2.98765, 0.00123]
    formatted_vector = format_vector(rotation_vector, decimal_places=2)
    print(formatted_vector)  # 出力: '1.12 -2.99 0.00'
    formatted_vector = format_vector(rotation_vector, decimal_places=2, format="decimal_exp")
    print(formatted_vector)  # 出力: '1.12 -2.99 0.00'
