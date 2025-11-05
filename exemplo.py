



loader = DirectWSGLoader(file_path=file_path)
wsg_obj = loader.load()
sklearn_pyg_data = DataConverter.to_pyg_data(wsg_obj, for_embedding_bag=False).to(DEVICE)


class PyTorchClassifier(BaseClassifier, nn.Module):
    """
    Classe base para classificadores PyTorch. Contém o loop de treino completo.
    """

    def __init__(
        self, config: Config, input_dim: int, hidden_dim: int, output_dim: int
    ):
        BaseClassifier.__init__(self, config)
        nn.Module.__init__(self)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def _train_step(self, optimizer, criterion, data, use_gnn):
        self.train()
        optimizer.zero_grad()

        args = [data.x, data.edge_index] if use_gnn else [data.x]
        out = self(*args)

        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def _test_step(self, data, use_gnn):
        self.eval()
        args = [data.x, data.edge_index] if use_gnn else [data.x]
        out = self(*args)
        pred = out.argmax(dim=1)

        y_true = data.y[data.test_mask]
        y_pred = pred[data.test_mask]

        acc = accuracy_score(y_true.cpu(), y_pred.cpu())
        f1 = f1_score(y_true.cpu(), y_pred.cpu(), average="weighted")
        report = classification_report(
            y_true.cpu(), y_pred.cpu(), output_dict=True, zero_division=0
        )

        return acc, f1, report

    def _train_and_evaluate_internal(
        self, data: Data, use_gnn: bool
    ):  # <-- MUDANÇA 3: Assinatura
        print(f"\n--- Avaliando (PyTorch): {self.model_name} ---")
        device = torch.device(self.config.DEVICE)
        self.to(device)

        # --- REMOVIDO ---
        # data = DataConverter.to_pyg_data(wsg_obj=wsg_obj, for_embedding_bag=False).to(device)
        # ---

        # 'data' agora é passado como argumento e já está no device

        optimizer = optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()

        start_time = time.process_time()

        pbar = tqdm(
            range(self.config.EPOCHS),
            desc=f"Treinando {self.model_name}",
            leave=False,
        )
        for epoch in pbar:
            loss = self._train_step(optimizer, criterion, data, use_gnn)
            pbar.set_postfix({"loss": f"{loss:.4f}"})

        train_time = time.process_time() - start_time

        acc, f1, report = self._test_step(data, use_gnn)
        return acc, f1, train_time, report


class MLPClassifier(PyTorchClassifier):
    """Classificador MLP que opera em um tensor de features denso."""

    def __init__(self, config, input_dim, hidden_dim, output_dim):
        super().__init__(config, input_dim, hidden_dim, output_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def train_and_evaluate(self, data: Data): 
        return self._train_and_evaluate_internal(data, use_gnn=False)


