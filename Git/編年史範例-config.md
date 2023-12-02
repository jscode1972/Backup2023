## git config -- global
個人設定
```
$ cat ~/.gitconfig
$ git config --list
$ git config --global user.name "Ben Huang"
$ git config --global user.email "jscode1972@gmail.com"
```

alias
```
$ git config --global alias.co checkout
$ git config --global alias.br branch
$ git config --global alias.st status
$ git config --global alias.l "log --oneline --graph"
$ git config --global alias.ls 'log --graph --pretty=format:"%h <%an> %ar %s"'
```

## git config --local 
專案層級
```
$ git config --local user.name "Ben Huang"              
$ git config --local user.email "jscode1972@gmail.com"
```

