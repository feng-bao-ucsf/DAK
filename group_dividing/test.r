setwd("/net/hsphs10/srv/export/hsphs10/share_root/hsphfs4/share_root/Christiani/mulong/Schizophrenia/AA/pathway_gene_for_plink")

rawlist <- read.table("raw_list.txt", header = FALSE)

for (i in as.character(rawlist$V1)){
  raw <- read.table(i, header = TRUE)
  raw[1:5, 1:10]
  
  phe <- raw[, 1:6]
  head(phe)
  
  snplist <- as.data.frame(colnames(raw[, 7:ncol(raw)]))
  head(snplist)

  geno <- raw[, 7:ncol(raw)]
  geno[1:5, 1:5]

  geno[geno=="0"] <- "100"
  geno[geno=="1"] <- "010"
  geno[geno=="2"] <- "001"
  geno[1:5, 1:5]

  f <- function(x){
    i1 <- substr(x, 1, 1)
    i2 <- substr(x, 2, 2)
    i3 <- substr(x, 3, 3)
    z <- data.frame(i1, i2, i3)
  }
  geno2 <- as.data.frame(apply(geno, 2, f))
  geno2[1:5, 1:6]

  write.table(phe, paste(i, "_phe.txt", sep = ""), col.names = TRUE, row.names = FALSE, quote = FALSE, sep = "\t")
  write.table(snplist, paste(i, "_snplist.txt", sep = ""), col.names = FALSE, row.names = FALSE, quote = FALSE, sep = "\t")
  write.table(geno2, paste(i, "_geno.txt", sep = ""), col.names = FALSE, row.names = FALSE, quote = FALSE, sep = "\t")
}
