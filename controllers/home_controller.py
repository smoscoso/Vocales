from views.backpropagation_info import BackpropagationInfoView

class HomeController:
    def __init__(self, root):
        """Inicializa el controlador de la vista de inicio"""
        # Inicializar la vista
        self.view = BackpropagationInfoView(root)
