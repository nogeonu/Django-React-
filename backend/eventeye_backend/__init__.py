try:
    import pymysql
    pymysql.install_as_MySQLdb()
except Exception:
    # PyMySQL이 설치되기 전 초기 import 에러를 무시
    pass

