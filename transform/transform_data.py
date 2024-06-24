from transform import db
from transform.traduzir.Tradutor import Tradutor

def transform_data():
    linhas = db.fetch_results("SELECT * FROM dados where texto_pt is NULL")
    for linha in linhas:
        dados = {
            "id": linha[0],
            "texto_en": linha[1]
        }

        try:
            tradutor = Tradutor(dados["texto_en"], "pt")
            texto_pt = tradutor.traduzir()
        except Exception as e:
            tradutor = Tradutor(dados["texto_en"], "pt")
            texto_pt = tradutor.traduzir()
        db.execute_query('''UPDATE public.dados SET texto_pt= %s WHERE id = %s;''',
                         params=(texto_pt, dados["id"]))
