import os


def create_directory_if_not_exists(directory: str):
    """
    指定したディレクトリが存在しない場合に作成します。

    Parameters:
    directory (str): 作成するディレクトリのパス
    """
    # import ipdb; ipdb.set_trace()
    try:
        os.makedirs(directory, exist_ok=True)
        # print(f"ディレクトリが存在しない場合は作成されました: {directory}")
    except Exception as e:
        print(f"ディレクトリの作成に失敗しました: {e}")
        raise
