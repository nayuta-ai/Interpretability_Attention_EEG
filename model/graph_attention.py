class GAT(torch.nn.Module):
    def __init__(self,num_features):
        self.conv = GATConv(num_features, 32)
    def forward(self,data):
        x, edge_index = data.x, data.edge_index
        x = self.conv(x,edge_index)
        return x