from googletrans import Translator


class Tradutor:
    def __init__(self, text, target_language):
        self.text = text
        self.target_language = target_language
        self._max_size = 1500
        self._min_size = 10
        self.translator = Translator()

    @property
    def max_size(self):
        return self._max_size

    @max_size.setter
    def max_size(self, size):
        if size > 6000:
            raise Exception("Tamanho máximo esperado 6000.")
        self._max_size = size

    @property
    def min_size(self):
        return self._min_size

    @min_size.setter
    def min_size(self, size):
        if size < 3:
            raise Exception("Tamanho mínimo esperado 3.")
        self._min_size = size

    def __get_size(self) -> int:
        return len(self.text)

    def __is_long(self):
        return self.__get_size() > self._max_size

    def __split_string(self, texto):
        partes = texto.split(".")
        resultado = []

        for parte in partes:
            while len(parte) > self._max_size:
                corte = parte.rfind(" ", 0, self._max_size)
                if corte == -1:
                    corte = self._max_size
                resultado.append((parte[:corte].strip(), False))
                parte = parte[corte:].strip()

            if resultado and len(parte) + len(resultado[-1][0]) <= self._max_size:
                ultima_parte, _ = resultado.pop()
                nova_parte = f"{ultima_parte}. {parte}".strip()
                resultado.append((nova_parte, True))
            elif len(parte) < self._min_size and resultado:
                ultima_parte, _ = resultado.pop()
                nova_parte = f"{ultima_parte}. {parte}".strip()
                resultado.append((nova_parte, True))
            else:
                resultado.append((parte.strip(), True))

        return [p for p in resultado if p[0]]

    def traduzir(self):
        retorno = ""
        if self.__is_long():
            for parte in self.__split_string(self.text):
                texto_traduzido = self.translator.translate(text=parte[0], dest=self.target_language)
                retorno += texto_traduzido.text
                if parte[1]:
                    retorno += "."
                else:
                    retorno += " "
        else:
            texto_traduzido = self.translator.translate(text=self.text, dest=self.target_language)
            retorno = texto_traduzido.text
        return retorno
