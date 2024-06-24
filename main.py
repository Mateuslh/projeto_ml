import argparse
import subprocess
from transform import transform_data
from modelo import treinar_modelo


def execute_command(command):
    try:
        result = subprocess.run(command, shell=True)
        if result.returncode == 0:
            print(f"Comando executado com sucesso:\n{result.stdout}")
        else:
            print(f"Erro ao executar comando:\n{result.stderr}")
    except Exception as e:
        print(f"Erro ao executar comando: {e}")


parser = argparse.ArgumentParser(description="Aplicação com fins de treinar um modelo e diponibilizar um endPoint para"
                                             " verificar a veracidade de noticas")

parser.add_argument('-t', action='store_true',
                    help='Treinar o modelo e disponibilizar para a api consumir.')
parser.add_argument('-r', action='store_true',
                    help='Iniciar a aplicação flask')
parser.add_argument('-a', action='store_true',
                    help='Alterar idioma do data set base')

args = parser.parse_args()

if args.t:
    treinar_modelo()
elif args.r:
    execute_command('python run.py')
elif args.a:
    transform_data()
else:
    print("Nenhum argumento fornecido. Use --cmd ou --function para especificar a ação.")
